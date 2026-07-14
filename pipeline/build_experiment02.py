"""Generate experiment02.ipynb -- R3-targeted oversampling on top of Exp 01."""
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Experiment 02 — class-balanced sampling for the rare grades (R2/R3)

Builds on **Experiment 01** (384 px + focal loss + macro-sensitivity selection). Exp 01
lifted R1 sensitivity (0.52→0.62) but **R3 stayed at 0.46** — because R3 is *data-limited*
(only ~124 R3 train eyes), not resolution-limited. This experiment attacks that directly.

## The single new lever (kept isolated for clean attribution)
**WeightedRandomSampler** — each training batch is drawn so the four grades appear
~equally often, instead of ~[57%, 33%, 5%, 5%]. The model now sees R2/R3 examples ~5–6×
more per epoch, which is the most direct way to raise their recall.

| | Exp 01 | **Exp 02** |
|---|---|---|
| input / backbone | 384 px ViT-L | 384 px ViT-L *(same)* |
| sampling | shuffle (natural freq) | **WeightedRandomSampler (class-balanced)** |
| loss | focal γ=2 **+ inverse-freq α-weights** | focal γ=2, **α-weights OFF** |
| selection | val macro-sensitivity | val macro-sensitivity *(same)* |

> **Why α-weights OFF:** a balanced sampler already corrects the imbalance by *frequency*.
> Keeping inverse-frequency loss weights on top would **double-correct** and over-suppress
> R0/R1 (tanking specificity). Sampler **or** class-weighted loss — not both.

> **Overfitting watch:** balancing means the ~276 R3 train images are shown ~5× more often,
> so they can be memorised. Mitigations already in the recipe: RandAug + random-erase + drop_path
> + weight decay + early best-checkpoint by val macro-sensitivity. `MINORITY_BOOST` below lets you
> push a grade *beyond* balance (e.g. R3×1.5) but raises this risk — leave at None first.
""")

md(r"""
## ⚠️ Gated model access — do this once
Accept <https://huggingface.co/YukunZhou/RETFound_mae_natureCFP> and
`huggingface-cli login --token <YOUR_HF_TOKEN>` (set `HF_ENDPOINT` mirror if HF is blocked).
""")

code(r"""
# --- ensure Phases 0-3 + cache + repo exist (skips if built) ---
import os, subprocess, sys
assert os.path.isdir("pipeline"), "Run from project root (Retfound.V2/)."
if not os.path.isdir("outputs/dr_imagefolder"):
    for s in ["build_manifest.py", "make_split.py", "materialize_imagefolder.py"]:
        print("running", s); subprocess.run([sys.executable, f"pipeline/{s}"], check=True)
if not os.path.isdir("outputs/dr_imagefolder_cache"):
    subprocess.run([sys.executable, "pipeline/build_resized_cache.py", "--size", "512"], check=True)
if not os.path.isdir("RETFound_repo"):
    subprocess.run(["git","clone","--depth","1","https://github.com/rmaphoh/RETFound.git","RETFound_repo"], check=True)
    subprocess.run([sys.executable,"-m","pip","install","-q","-r","RETFound_repo/requirements.txt"], check=True)
print("ready")
""")

code(r"""
# ============================ CONFIG (Experiment 02) ============================
CONFIG = dict(
    data_path   = "outputs/dr_imagefolder_cache",
    nb_classes  = 4,
    input_size  = 384,                 # from Exp 01
    finetune_id = "RETFound_mae_natureCFP",
    drop_path   = 0.2,
    adaptation  = "finetune",
    focal_gamma = 2.0,                 # focal kept; class weights handled by sampler instead

    # --- the new lever ---
    use_sampler   = True,              # WeightedRandomSampler (class-balanced batches)
    minority_boost = None,             # e.g. {3: 1.5} to push R3 beyond balance (overfit risk); None = pure balance

    batch_size    = 8, accum_iter = 8, # eff batch 64
    epochs        = 50, warmup_epochs = 10,
    blr = 5e-3, layer_decay = 0.65, weight_decay = 0.05, min_lr = 1e-6, clip_grad = None,
    device = "cuda", seed = 42, num_workers = 12,
    output_dir = "outputs/experiment02",
    task = "dr_exp02_384_focal_sampler",
)
SELECTION_METRIC = "macro_sensitivity"   # "balanced" | "qwk" | "macro_auroc" also available
import os; os.makedirs(CONFIG["output_dir"], exist_ok=True)
CONFIG
""")

code(r"""
# ============================ imports, seeds, device ============================
import os, sys, json, time, copy
import numpy as np, torch
sys.path.insert(0, "pipeline"); sys.path.insert(0, "RETFound_repo")
import dr_train as T, dr_eval as E
from dr_losses import FocalLoss
from engine_finetune import train_one_epoch

args = T.make_args(CONFIG)
T.set_seed(CONFIG["seed"]); torch.backends.cudnn.benchmark = True
device = torch.device(CONFIG["device"] if torch.cuda.is_available() else "cpu")
print("device:", device, "| input:", CONFIG["input_size"])
""")

code(r"""
# ============================ data + WeightedRandomSampler ============================
sampler = None
if CONFIG["use_sampler"]:
    # need the train dataset first to compute per-sample weights
    from util.datasets import build_dataset
    _ds = build_dataset(is_train="train", args=args)
    sampler, counts, exp_share = T.make_weighted_sampler(_ds, CONFIG["nb_classes"], CONFIG["minority_boost"])
    print("train class counts       :", counts)
    print("expected per-class share/batch:", np.round(exp_share, 3), "(≈ balanced)")

(ds_tr, ds_va, ds_te), (dl_tr, dl_va, dl_te) = T.build_loaders(args, train_sampler=sampler)
assert ds_tr.class_to_idx == json.load(open("outputs/class_mapping.json"))["ordinal_class_to_index"]
print("images train/val/test:", len(ds_tr), len(ds_va), len(ds_te), "| sampler:", sampler is not None)
""")

code(r"""
# ============================ model + gated weights ============================
model = T.build_model_arch(args)
msg = T.load_pretrained(model, args)
model.to(device)
print("missing keys (expect head.* + fc_norm.*):", list(msg.missing_keys))
print(f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f} M")
""")

code(r"""
# ============================ optimizer + criterion ============================
# Sampler balances frequency -> use focal WITHOUT class weights (avoid double-correction).
optimizer, loss_scaler = T.build_optimizer(model, args)
if CONFIG["use_sampler"]:
    criterion = FocalLoss(weight=None, gamma=CONFIG["focal_gamma"])
    print("criterion: FocalLoss (gamma only, NO class weights — sampler handles imbalance)")
else:
    cw, _ = T.class_weights_from_dataset(ds_tr, CONFIG["nb_classes"], device)
    criterion = FocalLoss(weight=cw, gamma=CONFIG["focal_gamma"])
    print("criterion: FocalLoss (gamma + inverse-freq class weights)")
print(f"param groups: {len(optimizer.param_groups)} | base lr: {args.lr:.2e} | eff batch: {args.batch_size*args.accum_iter}")
""")

code(r"""
# ============================ training loop (select by val macro-sensitivity) ============================
from sklearn.metrics import cohen_kappa_score, roc_auc_score

def val_scores():
    y, p = E.predict(model, dl_va, device)
    pred = p.argmax(1)
    qwk = cohen_kappa_score(y, pred, weights="quadratic", labels=list(range(CONFIG["nb_classes"])))
    try:
        yoh = np.eye(CONFIG["nb_classes"])[y]; cols = [c for c in range(CONFIG["nb_classes"]) if yoh[:,c].sum()>0]
        auroc = roc_auc_score(yoh[:,cols], p[:,cols], average="macro", multi_class="ovr")
    except Exception: auroc = float("nan")
    msens, mspec = E.macro_sens_spec(y, pred)
    return float(qwk), float(auroc), msens, mspec

def selection_score(qwk, auroc, msens, mspec):
    return {"macro_sensitivity": msens, "balanced": 0.5*(msens+mspec),
            "qwk": qwk, "macro_auroc": auroc}[SELECTION_METRIC]

best_score, best_epoch, history = -1.0, -1, []
ckpt_path = os.path.join(CONFIG["output_dir"], "checkpoint-best.pth")
t0 = time.time()
for epoch in range(CONFIG["epochs"]):
    tr = train_one_epoch(model, criterion, dl_tr, optimizer, device, epoch,
                         loss_scaler, args.clip_grad, None, None, args)
    qwk, auroc, msens, mspec = val_scores()
    score = selection_score(qwk, auroc, msens, mspec)
    history.append({"epoch": epoch, "train_loss": tr["loss"], "val_qwk": qwk, "val_macro_auroc": auroc,
                    "val_macro_sensitivity": msens, "val_macro_specificity": mspec, "selection_score": score})
    tag = ""
    if score > best_score:
        best_score, best_epoch = score, epoch
        torch.save({"model": copy.deepcopy(model.state_dict()), "epoch": epoch, "config": CONFIG,
                    "selection_metric": SELECTION_METRIC, "val_qwk": qwk, "val_macro_auroc": auroc,
                    "val_macro_sensitivity": msens, "val_macro_specificity": mspec}, ckpt_path)
        tag = "  <-- best"
    print(f"epoch {epoch:02d}  loss={tr['loss']:.4f}  val_QWK={qwk:.4f}  AUROC={auroc:.4f}  "
          f"mSens={msens:.4f}  mSpec={mspec:.4f}{tag}")
json.dump(history, open(os.path.join(CONFIG["output_dir"], "history.json"), "w"), indent=2)
print(f"\nDone in {(time.time()-t0)/60:.1f} min. Best epoch {best_epoch}  {SELECTION_METRIC}={best_score:.4f}")
""")

code(r"""
# ---- training curves ----
import matplotlib.pyplot as plt
h = history; ep = [x["epoch"] for x in h]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(ep, [x["train_loss"] for x in h]); ax[0].set_title("train loss"); ax[0].set_xlabel("epoch")
for k, lab in [("val_qwk","QWK"),("val_macro_auroc","macro-AUROC"),
               ("val_macro_sensitivity","macro-sensitivity"),("val_macro_specificity","macro-specificity")]:
    ax[1].plot(ep, [x[k] for x in h], label=lab)
ax[1].axvline(best_epoch, ls="--", c="grey"); ax[1].legend(fontsize=8)
ax[1].set_title(f"validation (selected by {SELECTION_METRIC})"); ax[1].set_xlabel("epoch")
fig.tight_layout(); plt.show()
""")

md(r"""
## Quick test-set check
Full evaluation: **`evaluation/experiment02_evaluation.ipynb`**. Quick eye-level summary below,
focus on **R2 / R3 sensitivity** vs Exp 01 (R2 0.77, R3 0.46).
""")

code(r"""
# ============================ quick eval on TEST (best checkpoint) ============================
best = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(best["model"]); model.to(device)
print(f"Best epoch {best['epoch']} | val macro-sens {best['val_macro_sensitivity']:.4f}")
test_paths = [p for p, _ in ds_te.samples]
y_true, y_prob = E.predict(model, dl_te, device)
rep = E.full_report(test_paths, y_true, y_prob, os.path.join(CONFIG["output_dir"], "eval_test"))
r = rep["eye_level"]; b = r["binary_referable"]
print(f"\nEYE-LEVEL (n={r['n']}): QWK={r['qwk']:.4f}  macroAUROC={r['macro_auroc_ovr']:.4f}  "
      f"macro_sens={r['macro_sensitivity']:.4f}  macro_spec={r['macro_specificity']:.4f}")
print("per-class sens/spec:", {['R0','R1','R2','R3'][k]: (round(v['sensitivity'],3), round(v['specificity'],3))
                               for k, v in r['per_class'].items()})
print(f"referable-DR: AUROC={b['auroc']:.4f} sens={b['sensitivity']:.3f} spec={b['specificity']:.3f} @0.5")
""")

md(r"""
## Compare (eye level)
| metric | Exp 01 (384+focal+wCE) | **Exp 02 (+ balanced sampler)** |
|---|---|---|
| macro-sensitivity | 0.679 | _(fill)_ |
| **R2 / R3 sensitivity** | 0.77 / 0.46 | _(fill — the target)_ |
| macro-specificity | 0.901 | _(fill — watch for drop)_ |
| QWK | 0.737 | _(fill)_ |
| referable sens/spec @0.5 | 0.825 / 0.941 | _(fill)_ |

Expect R2/R3 sensitivity **up** and specificity **down** (the balance tradeoff). If R3 still
doesn't move, it's genuinely data-starved → next levers: recover the 798 discordant images
(more R3) or k-fold CV to see if the effect is even distinguishable from noise (R3 test n≈22).
""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "retfound", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiment02.ipynb")
nbf.write(nb, out)
print("wrote", out, "with", len(cells), "cells")
