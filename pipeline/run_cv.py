"""
K-fold cross-validation runner (patient-level).

Prereq: `python pipeline/make_split.py --kfolds 5` (writes a `fold` column to the manifest).

For each fold f: test = fold f; a stratified ~15% (by patient worst-grade) of the remaining
folds is held out as val; the rest is train. Materializes symlink ImageFolders under
outputs/cv/foldN/, trains the RETFound recipe (same as the experiments), evaluates eye-level
on the test fold, and aggregates mean±std across folds -> outputs/cv/cv_results.json.

This trains K models (~K x single-run time). Use --folds to run a subset.

Example:
  python pipeline/run_cv.py --kfolds 5 --input-size 384 --use-sampler --epochs 50
"""
import os
import sys
import json
import shutil
import argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import dr_train as T
import dr_eval as E
T._ensure_repo_on_path()   # put RETFound_repo on sys.path so `engine_finetune` imports
from sklearn.model_selection import train_test_split

CV_ROOT = os.path.join(C.OUT_DIR, "cv")


def sanitize(s):
    import re
    return re.sub(r"[^0-9A-Za-z._-]", "", str(s).replace(" ", "_"))


CACHE_ROOT = os.path.join(C.OUT_DIR, "dr_imagefolder_cache")


def _cache_src(r):
    """Deterministic path to a row's image in the clean 512px cache (organised by the
    ORIGINAL train/val/test split). Falls back to the 12MP original if not cached."""
    name = sanitize(f"{r['patient_id']}_{r['eye']}_{os.path.basename(r['image_path'])}")
    cached = os.path.join(CACHE_ROOT, r["split"], C.ORDINAL_CLASS_NAMES[int(r["dr_label"])], name)
    return cached if os.path.exists(cached) else os.path.abspath(r["image_path"])


def materialize_fold(df_rows, split_of_patient, root):
    """Build root/{train,val,test}/<class>/<pid>_<eye>_<orig> symlinks -> clean 512px cache."""
    if os.path.isdir(root):
        shutil.rmtree(root)
    for _, r in df_rows.iterrows():
        split = split_of_patient.get(r["patient_id"])
        if split is None:
            continue
        cls = C.ORDINAL_CLASS_NAMES[int(r["dr_label"])]
        d = os.path.join(root, split, cls); os.makedirs(d, exist_ok=True)
        link = os.path.join(d, sanitize(f"{r['patient_id']}_{r['eye']}_{os.path.basename(r['image_path'])}"))
        if not os.path.exists(link):
            os.symlink(_cache_src(r), link)


def train_one_fold(root, cfg, device):
    from engine_finetune import train_one_epoch
    from dr_losses import FocalLoss
    args = T.make_args({**cfg, "data_path": root, "output_dir": os.path.join(root, "_out")})
    T.set_seed(cfg["seed"])
    sampler = None
    if cfg.get("use_sampler"):
        from util.datasets import build_dataset
        _ds = build_dataset(is_train="train", args=args)
        sampler, _, _ = T.make_weighted_sampler(_ds, cfg["nb_classes"], cfg.get("minority_boost"))
    (ds_tr, ds_va, ds_te), (dl_tr, dl_va, dl_te) = T.build_loaders(args, train_sampler=sampler)
    model = T.build_model_arch(args); T.load_pretrained(model, args); model.to(device)
    optimizer, scaler = T.build_optimizer(model, args)
    if sampler is not None:
        criterion = FocalLoss(weight=None, gamma=cfg["focal_gamma"])
    else:
        cw, _ = T.class_weights_from_dataset(ds_tr, cfg["nb_classes"], device)
        criterion = FocalLoss(weight=cw, gamma=cfg["focal_gamma"])

    best, best_state = -1.0, None
    for epoch in range(cfg["epochs"]):
        train_one_epoch(model, criterion, dl_tr, optimizer, device, epoch, scaler,
                        args.clip_grad, None, None, args)
        y, p = E.predict(model, dl_va, device)
        msens, _ = E.macro_sens_spec(y, p.argmax(1))
        if msens > best:
            best, best_state = msens, {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    test_paths = [p for p, _ in ds_te.samples]
    y, p = E.predict(model, dl_te, device, tta=cfg.get("tta"))
    yt, yp = E.aggregate(test_paths, y, p, "eye")
    return E.compute_metrics(yt, yp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kfolds", type=int, required=True)
    ap.add_argument("--input-size", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accum-iter", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--use-sampler", action="store_true")
    ap.add_argument("--folds", default=None, help="comma list subset, e.g. 0,1")
    ap.add_argument("--tta", default=None, help="comma list of TTA views, e.g. identity,hflip,vflip,hvflip")
    args = ap.parse_args()

    m = pd.read_csv(C.MANIFEST_PATH)
    assert "fold" in m.columns, "run `make_split.py --kfolds K` first"
    u = m[(m["usable"] == True) & (m["fold"] != "")].copy()  # noqa: E712
    u["dr_label"] = u["dr_label"].astype(int); u["fold"] = u["fold"].astype(float).astype(int)

    cfg = dict(nb_classes=4, input_size=args.input_size, finetune_id="RETFound_mae_natureCFP",
               drop_path=0.2, focal_gamma=2.0, use_sampler=args.use_sampler, minority_boost=None,
               batch_size=args.batch_size, accum_iter=args.accum_iter, epochs=args.epochs,
               warmup_epochs=max(1, args.epochs // 5), blr=5e-3, layer_decay=0.65, weight_decay=0.05,
               min_lr=1e-6, clip_grad=None, device="cuda", seed=42, num_workers=12,
               tta=(args.tta.split(",") if args.tta else None))
    device = torch.device("cuda")
    folds = [int(x) for x in args.folds.split(",")] if args.folds else list(range(args.kfolds))

    os.makedirs(CV_ROOT, exist_ok=True)
    results = {}
    for f in folds:
        print(f"\n########## FOLD {f}/{args.kfolds - 1} ##########")
        test_pat = set(u[u["fold"] == f]["patient_id"])
        rest = u[u["fold"] != f]
        rest_pat = rest.groupby("patient_id")["dr_label"].max().reset_index()
        tr_pat, va_pat = train_test_split(rest_pat, test_size=0.1765, random_state=42,
                                          stratify=rest_pat["dr_label"])
        split_of = {p: "train" for p in tr_pat["patient_id"]}
        split_of.update({p: "val" for p in va_pat["patient_id"]})
        split_of.update({p: "test" for p in test_pat})
        root = os.path.join(CV_ROOT, f"fold{f}")
        materialize_fold(u, split_of, root)
        res = train_one_fold(root, cfg, device)
        results[f] = res
        print(f"fold {f}: QWK={res['qwk']:.4f} mSens={res['macro_sensitivity']:.4f} "
              f"mSpec={res['macro_specificity']:.4f} refAUROC={res['binary_referable'].get('auroc')}")
        json.dump(results, open(os.path.join(CV_ROOT, "cv_results.json"), "w"), indent=2)

    # aggregate
    keys = ["qwk", "macro_auroc_ovr", "macro_sensitivity", "macro_specificity", "accuracy"]
    print("\n===== CV SUMMARY (mean ± std across folds) =====")
    agg = {}
    for k in keys:
        vals = [results[f][k] for f in results]
        agg[k] = [float(np.mean(vals)), float(np.std(vals))]
        print(f"  {k:20}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    ref = [results[f]["binary_referable"].get("auroc") for f in results if results[f]["binary_referable"].get("auroc")]
    if ref:
        agg["referable_auroc"] = [float(np.mean(ref)), float(np.std(ref))]
        print(f"  {'referable_auroc':20}: {np.mean(ref):.4f} ± {np.std(ref):.4f}")
    json.dump({"per_fold": results, "aggregate": agg}, open(os.path.join(CV_ROOT, "cv_results.json"), "w"), indent=2)
    print(f"\nSaved {CV_ROOT}/cv_results.json")


if __name__ == "__main__":
    main()
