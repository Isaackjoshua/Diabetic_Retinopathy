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


def _build_criterion(cfg, ds_tr, device):
    """Pick the training loss. `loss` in cfg: 'focal' (default) | 'logit_adjusted'."""
    from dr_losses import FocalLoss, LogitAdjustedLoss
    import numpy as _np
    counts = _np.bincount(_np.array(ds_tr.targets), minlength=cfg["nb_classes"])
    if cfg.get("loss") == "logit_adjusted":
        # logit adjustment already handles imbalance -> no inverse-freq weight
        return LogitAdjustedLoss(counts, tau=cfg.get("la_tau", 1.0))
    if cfg.get("use_sampler"):
        return FocalLoss(weight=None, gamma=cfg["focal_gamma"])
    cw, _ = T.class_weights_from_dataset(ds_tr, cfg["nb_classes"], device)
    return FocalLoss(weight=cw, gamma=cfg["focal_gamma"])


def train_one_fold(root, cfg, device):
    from engine_finetune import train_one_epoch
    args = T.make_args({**cfg, "data_path": root, "output_dir": os.path.join(root, "_out")})
    T.set_seed(cfg["seed"])
    sampler = None
    if cfg.get("use_sampler") and cfg.get("loss") != "logit_adjusted":
        from util.datasets import build_dataset
        _ds = build_dataset(is_train="train", args=args)
        sampler, _, _ = T.make_weighted_sampler(_ds, cfg["nb_classes"], cfg.get("minority_boost"))
    (ds_tr, ds_va, ds_te), (dl_tr, dl_va, dl_te) = T.build_loaders(args, train_sampler=sampler)
    model = T.build_model_arch(args); T.load_pretrained(model, args); model.to(device)
    optimizer, scaler = T.build_optimizer(model, args)
    criterion = _build_criterion(cfg, ds_tr, device)

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
    yt, yp = E.aggregate(test_paths, y, p, "eye")     # eye-level OOF preds for this fold
    return E.compute_metrics(yt, yp), np.asarray(yt).astype(int), np.asarray(yp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kfolds", type=int, required=True)
    ap.add_argument("--input-size", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accum-iter", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--use-sampler", action="store_true")
    ap.add_argument("--backbone", choices=["mae", "dinov2"], default="mae")
    ap.add_argument("--loss", choices=["focal", "logit_adjusted"], default="focal")
    ap.add_argument("--la-tau", type=float, default=1.0, help="logit-adjustment strength (loss=logit_adjusted)")
    ap.add_argument("--folds", default=None, help="comma list subset, e.g. 0,1")
    ap.add_argument("--tta", default=None, help="comma list of TTA views, e.g. identity,hflip,vflip,hvflip")
    args = ap.parse_args()

    m = pd.read_csv(C.MANIFEST_PATH)
    assert "fold" in m.columns, "run `make_split.py --kfolds K` first"
    u = m[(m["usable"] == True) & (m["fold"] != "")].copy()  # noqa: E712
    u["dr_label"] = u["dr_label"].astype(int); u["fold"] = u["fold"].astype(float).astype(int)

    # backbone-specific model wiring (mirrors build_finetune_dinov2 / RETFound_mae)
    if args.backbone == "dinov2":
        bb = dict(model="RETFound_dinov2", model_arch="dinov2_vitl14",
                  finetune_id="RETFound_dinov2_meh", input_size=224)   # DINOv2 fixes img 224
    else:
        bb = dict(model="RETFound_mae", model_arch="retfound_mae",
                  finetune_id="RETFound_mae_natureCFP", input_size=args.input_size)

    cfg = dict(nb_classes=4, drop_path=0.2, focal_gamma=2.0,
               loss=args.loss, la_tau=args.la_tau,
               use_sampler=args.use_sampler, minority_boost=None,
               batch_size=args.batch_size, accum_iter=args.accum_iter, epochs=args.epochs,
               warmup_epochs=max(1, args.epochs // 5), blr=5e-3, layer_decay=0.65, weight_decay=0.05,
               min_lr=1e-6, clip_grad=None, device="cuda", seed=42, num_workers=12,
               tta=(args.tta.split(",") if args.tta else None), **bb)
    device = torch.device("cuda")
    folds = [int(x) for x in args.folds.split(",")] if args.folds else list(range(args.kfolds))

    os.makedirs(CV_ROOT, exist_ok=True)
    results = {}
    oof_y, oof_p = [], []          # pooled out-of-fold eye-level preds across folds
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
        res, yt, yp = train_one_fold(root, cfg, device)
        results[f] = res
        oof_y.append(yt); oof_p.append(yp)
        np.savez(os.path.join(CV_ROOT, f"oof_fold{f}.npz"), y=yt, p=yp)   # persist for re-analysis
        print(f"fold {f}: mSens={res['macro_sensitivity']:.4f} mSpec={res['macro_specificity']:.4f} "
              f"macroAUROC={res['macro_auroc_ovr']:.4f} refAUROC={res['binary_referable'].get('auroc')}")
        json.dump({int(k): v for k, v in results.items()},
                  open(os.path.join(CV_ROOT, "cv_results.json"), "w"), indent=2)

    # ---- per-fold aggregate (mean ± std) ----
    keys = ["macro_sensitivity", "macro_specificity", "macro_auroc_ovr", "accuracy", "qwk"]
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

    # ---- POOLED out-of-fold: the stable macro-sensitivity estimate + bootstrap 95% CI ----
    pool = None
    if oof_y:
        Y = np.concatenate(oof_y); P = np.concatenate(oof_p)
        pooled = E.compute_metrics(Y, P)
        rng = np.random.default_rng(42); n = len(Y); boot = []
        for _ in range(2000):
            idx = rng.integers(0, n, n)
            ms, _ = E.macro_sens_spec(Y[idx], P[idx].argmax(1))
            boot.append(ms)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        per_cls = {C.ORDINAL_CLASS_NAMES[int(c)]: round(v["sensitivity"], 3)
                   for c, v in pooled["per_class"].items()}
        pool = dict(n_eyes=int(n), macro_sensitivity=pooled["macro_sensitivity"],
                    macro_sensitivity_ci95=[float(lo), float(hi)],
                    macro_specificity=pooled["macro_specificity"],
                    macro_auroc_ovr=pooled["macro_auroc_ovr"],
                    referable_auroc=pooled["binary_referable"].get("auroc"),
                    per_class_sensitivity=per_cls)
        print("\n===== POOLED OUT-OF-FOLD (all folds concatenated; the number to trust) =====")
        print(f"  eyes pooled           : {n}")
        print(f"  MACRO-SENSITIVITY     : {pooled['macro_sensitivity']:.4f}  "
              f"(95% CI {lo:.3f}-{hi:.3f})   <-- target > 0.80")
        print(f"  per-class sensitivity : {per_cls}")
        print(f"  macro-specificity     : {pooled['macro_specificity']:.4f}")
        print(f"  macro-AUROC / refAUROC: {pooled['macro_auroc_ovr']:.4f} / {pooled['binary_referable'].get('auroc')}")

    cfg_saved = {k: cfg[k] for k in ("model", "model_arch", "input_size", "loss", "la_tau",
                                     "use_sampler", "epochs", "batch_size", "accum_iter", "tta")}
    out = dict(config=cfg_saved, per_fold={int(k): v for k, v in results.items()},
               aggregate=agg, pooled_oof=pool)
    json.dump(out, open(os.path.join(CV_ROOT, "cv_results.json"), "w"), indent=2)
    print(f"\nSaved {CV_ROOT}/cv_results.json")


if __name__ == "__main__":
    main()
