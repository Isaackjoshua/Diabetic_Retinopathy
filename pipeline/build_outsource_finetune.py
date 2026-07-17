"""Generate Outsource-finetune.ipynb -- option #1: external-data transfer to enrich rare
grades. Two-phase DINOv2: Phase 1 pre-fine-tune on APTOS-2019 (large public DR set, many
R2/R3), Phase 2 fine-tune on the LOCAL set. Selection + all reported metrics are on LOCAL
val/test -- the domain that matters. Compared against the exp03 pooled-OOF ceiling (0.765)."""
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Outsource-finetune — external-data transfer (APTOS-2019 → local) to lift rare grades

**Why:** every *in-distribution* lever (reweighting, focal, sampler, logit adjustment,
threshold tuning) capped 4-class **macro-sensitivity at ~0.765** (exp03, 5-fold CV pooled OOF,
95% CI 0.741–0.788). The binding constraint is data scarcity + low appearance diversity in the
rare grades (local train ≈ 142 R2 / 124 R3 *eyes*). No reweighting invents signal that isn't
there — so we add **real external information**.

**APTOS-2019** is a large public colour-fundus DR set graded ICDR 0–4. Mapped onto our NHS
4-class scheme (`build_aptos_imagefolder.py`), it adds **~1,192 R2 and ~295 R3 images** —
roughly 4× the R2 and 2× the R3 the local set has.

### Two-phase recipe (order matters)
| phase | data | init | loss | selects on |
|---|---|---|---|---|
| **1 — external pre-fine-tune** | APTOS train | RETFound DINOv2 (teacher) | weighted-CE | APTOS val macro-sens |
| **2 — local fine-tune** | LOCAL train | **Phase-1 checkpoint** | logit-adjusted (τ=1.0) | LOCAL val macro-sens |

Phase 1 teaches general DR features (incl. abundant R2/R3); Phase 2 domain-adapts to *your*
cameras/population. We fine-tune on local **after** APTOS so the external distribution doesn't
dominate — and we select/evaluate only on local data. The 4-way head transfers between phases
because `build_aptos_imagefolder.py` uses the SAME class indices as the local ImageFolder.

### Honest framing
This is a **single-split** run (fast, directional): does the APTOS init lift local test
macro-sensitivity above the ~0.715 single-split baseline / 0.765 CV ceiling? If yes, confirm
with 5-fold CV (Section C) before any claim — single-split macro-sens swings ~0.10 on this data.

### Laterality / data unchanged: OD=RE, OS=LE; local 8407 img / 3834 eyes / 2194 patients.
""")

code(r"""
# --- ensure local Phases 0-3 + cache + RETFound repo + APTOS ImageFolder exist (skip if built) ---
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
if not os.path.isdir("outputs/aptos_imagefolder"):
    print("materializing APTOS ImageFolder..."); subprocess.run([sys.executable, "pipeline/build_aptos_imagefolder.py"], check=True)
print("ready")
""")

code(r"""
# ============================ CONFIG ============================
BASE = dict(
    nb_classes = 4, input_size = 224,             # DINOv2 fixes img_size=224 (patch14)
    model = "RETFound_dinov2", model_arch = "dinov2_vitl14",
    finetune_id = "RETFound_dinov2_meh",          # GATED HF weights (checkpoint key: "teacher")
    drop_path = 0.2, adaptation = "finetune",
    batch_size = 8, accum_iter = 8,               # eff batch 64 (bs8 to fit 16GB safely)
    blr = 5e-3, layer_decay = 0.65, weight_decay = 0.05, min_lr = 1e-6, clip_grad = None,
    device = "cuda", seed = 42, num_workers = 10,
)
OUT = "outputs/outsource_finetune"; os.makedirs(OUT, exist_ok=True)
# phase 1: APTOS external pre-fine-tune
CFG_APTOS = dict(BASE, data_path="outputs/aptos_imagefolder", output_dir=f"{OUT}/phase1_aptos",
                 epochs=30, warmup_epochs=6, task="aptos_pretrain", loss="weighted_ce")
# phase 2: local fine-tune (logit-adjusted, our validated recipe)
CFG_LOCAL = dict(BASE, data_path="outputs/dr_imagefolder_cache", output_dir=OUT,
                 epochs=50, warmup_epochs=10, task="local_finetune", loss="logit_adjusted", la_tau=1.0)
SELECTION_METRIC = "macro_sensitivity"
import os
for c in (CFG_APTOS, CFG_LOCAL): os.makedirs(c["output_dir"], exist_ok=True)
print("phase1 (APTOS):", CFG_APTOS["output_dir"], "| phase2 (local):", CFG_LOCAL["output_dir"])
""")

code(r"""
# ============================ imports, seeds, device + shared trainer ============================
import os, sys, json, time, copy
import numpy as np, torch
sys.path.insert(0, "pipeline"); sys.path.insert(0, "RETFound_repo")
import dr_train as T, dr_eval as E
from dr_losses import LogitAdjustedLoss
from engine_finetune import train_one_epoch
from sklearn.metrics import roc_auc_score

T.set_seed(BASE["seed"]); torch.backends.cudnn.benchmark = True
device = torch.device(BASE["device"] if torch.cuda.is_available() else "cpu")
print("device:", device, "| backbone:", BASE["model"])

def make_criterion(cfg, ds_tr):
    counts = np.bincount(np.array(ds_tr.targets), minlength=cfg["nb_classes"])
    if cfg["loss"] == "logit_adjusted":
        return LogitAdjustedLoss(counts, tau=cfg.get("la_tau", 1.0)), counts   # raw logits at inference
    cw, _ = T.class_weights_from_dataset(ds_tr, cfg["nb_classes"], device)      # weighted CE
    return torch.nn.CrossEntropyLoss(weight=cw), counts

def val_scores(model, dl_va):
    y, p = E.predict(model, dl_va, device); pred = p.argmax(1)
    try:
        yoh = np.eye(BASE["nb_classes"])[y]; cols=[c for c in range(BASE["nb_classes"]) if yoh[:,c].sum()>0]
        auroc = roc_auc_score(yoh[:,cols], p[:,cols], average="macro", multi_class="ovr")
    except Exception: auroc = float("nan")
    msens, mspec = E.macro_sens_spec(y, pred)
    return float(auroc), msens, mspec

def train_phase(model, cfg, dl_tr, dl_va, criterion, tag):
    args = T.make_args(cfg)                            # ONE args: build_optimizer sets args.lr on it
    optimizer, scaler = T.build_optimizer(model, args) # (make_args leaves lr=None until here)
    best, best_ep, hist = -1.0, -1, []
    ckpt = os.path.join(cfg["output_dir"], "checkpoint-best.pth"); t0 = time.time()
    for epoch in range(cfg["epochs"]):
        tr = train_one_epoch(model, criterion, dl_tr, optimizer, device, epoch, scaler,
                             args.clip_grad, None, None, args)
        auroc, msens, mspec = val_scores(model, dl_va)
        hist.append({"epoch": epoch, "train_loss": tr["loss"], "val_macro_auroc": auroc,
                     "val_macro_sensitivity": msens, "val_macro_specificity": mspec})
        flag = ""
        if msens > best:
            best, best_ep = msens, epoch
            torch.save({"model": copy.deepcopy(model.state_dict()), "epoch": epoch, "config": cfg,
                        "val_macro_sensitivity": msens, "val_macro_specificity": mspec,
                        "val_macro_auroc": auroc}, ckpt); flag = "  <-- best"
        print(f"[{tag}] ep {epoch:02d} loss={tr['loss']:.4f} val_mSens={msens:.4f} "
              f"val_mSpec={mspec:.4f} val_AUROC={auroc:.4f}{flag}")
    json.dump(hist, open(os.path.join(cfg["output_dir"], "history.json"), "w"), indent=2)
    print(f"[{tag}] done in {(time.time()-t0)/60:.1f} min | best epoch {best_ep} val_mSens={best:.4f}")
    return ckpt, hist
""")

md(r"""
## Phase 1 — pre-fine-tune on APTOS-2019 (external)
Starts from RETFound DINOv2 teacher weights, learns general DR features from the large public
set (abundant R2/R3), selected on APTOS val macro-sensitivity. ~30 epochs on 2,930 images.
""")

code(r"""
# ---- Phase 1 data + model + train ----
args_a = T.make_args(CFG_APTOS)
(a_tr, a_va, a_te), (adl_tr, adl_va, adl_te) = T.build_loaders(args_a, shuffle_train=True)
print("APTOS images train/val/test:", len(a_tr), len(a_va), len(a_te))
assert a_tr.class_to_idx == json.load(open("outputs/class_mapping.json"))["ordinal_class_to_index"], \
    "APTOS class indices must match local"

model = T.build_model_arch(args_a); msg = T.load_pretrained(model, args_a); model.to(device)
print("missing keys (expect head.* only):", list(msg.missing_keys))
crit_a, counts_a = make_criterion(CFG_APTOS, a_tr)
print("APTOS train class counts:", counts_a)
phase1_ckpt, _ = train_phase(model, CFG_APTOS, adl_tr, adl_va, crit_a, "APTOS")
""")

md(r"""
## Phase 2 — fine-tune on the LOCAL set
Initialise from the Phase-1 checkpoint (features + 4-way head already DR-aware), then
domain-adapt to the local cameras/population with the logit-adjusted loss, selected on LOCAL
val macro-sensitivity.
""")

code(r"""
# ---- Phase 2 data + init from Phase 1 + train ----
args_l = T.make_args(CFG_LOCAL)
(l_tr, l_va, l_te), (ldl_tr, ldl_va, ldl_te) = T.build_loaders(args_l, shuffle_train=True)
print("LOCAL images train/val/test:", len(l_tr), len(l_va), len(l_te))

model = T.build_model_arch(args_l)                       # fresh DINOv2 arch
sd = torch.load(phase1_ckpt, map_location="cpu")["model"]
miss, unexp = model.load_state_dict(sd, strict=False)    # head shapes match -> transfers too
model.to(device)
print(f"loaded Phase-1 weights | missing={len(miss)} unexpected={len(unexp)}")
crit_l, counts_l = make_criterion(CFG_LOCAL, l_tr)
print("LOCAL train class counts:", counts_l, "| loss:", CFG_LOCAL["loss"], "tau", CFG_LOCAL.get("la_tau"))
local_ckpt, hist_l = train_phase(model, CFG_LOCAL, ldl_tr, ldl_va, crit_l, "LOCAL")
""")

code(r"""
# ---- training curves (Phase 2) ----
import matplotlib.pyplot as plt
ep = [x["epoch"] for x in hist_l]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(ep, [x["train_loss"] for x in hist_l]); ax[0].set_title("Phase 2 train loss"); ax[0].set_xlabel("epoch")
for k, lab in [("val_macro_sensitivity","macro-sensitivity"),
               ("val_macro_specificity","macro-specificity"),("val_macro_auroc","macro-AUROC")]:
    ax[1].plot(ep, [x[k] for x in hist_l], label=lab)
ax[1].axhline(0.80, ls=":", c="red", label="target 0.80")
ax[1].axhline(0.765, ls="--", c="orange", label="exp03 CV ceiling")
ax[1].legend(fontsize=8); ax[1].set_title("Phase 2 validation (local)"); ax[1].set_xlabel("epoch")
fig.tight_layout(); plt.show()
""")

md(r"""
## Evaluate on the LOCAL test set + compare to the ceiling
Eye-level macro-sensitivity with TTA, vs the exp03 in-distribution ceiling (pooled-OOF 0.765).
""")

code(r"""
# ---- LOCAL test (best Phase-2 checkpoint) ----
best = torch.load(local_ckpt, map_location="cpu")
model.load_state_dict(best["model"]); model.to(device)
test_paths = [p for p, _ in l_te.samples]
y_true, y_prob = E.predict(model, ldl_te, device, tta=["identity","hflip","vflip","hvflip"])
rep = E.full_report(test_paths, y_true, y_prob, os.path.join(OUT, "eval_test"))
r = rep["eye_level"]; b = r["binary_referable"]
print(f"EYE-LEVEL (n={r['n']}): MACRO-SENS={r['macro_sensitivity']:.4f}  "
      f"mSpec={r['macro_specificity']:.4f}  macroAUROC={r['macro_auroc_ovr']:.4f}")
print("per-class sens:", {['R0','R1','R2','R3'][k]: round(v['sensitivity'],3) for k,v in r['per_class'].items()})
print(f"referable(R2+): AUROC={b['auroc']:.4f} sens={b['sensitivity']:.3f} spec={b['specificity']:.3f}")
print("\ncompare (eye-level test macro-sens):")
print(f"  DINOv2 logit-adj, NO APTOS (exp03 single-split): 0.715   | exp03 5-fold CV pooled: 0.765")
print(f"  DINOv2 logit-adj, + APTOS  (this run)          : {r['macro_sensitivity']:.3f}")
print("\n⚠️  single split — swings ~0.10. If this beats 0.765, confirm with Section C (5-fold CV).")
""")

md(r"""
## Section C — confirm with 5-fold CV (only if Section A/B beat the ceiling)
The single-split number above is directional. To make a defensible claim, repeat the *local*
phase under 5-fold CV, initialising every fold from the **same** Phase-1 APTOS checkpoint
(Phase 1 is domain-external, so it need not be refit per fold — reuse `phase1_ckpt`).

`run_cv.py` supports an external-init via `--init-ckpt`; run:
```bash
python pipeline/run_cv.py --kfolds 5 --backbone dinov2 --loss logit_adjusted --la-tau 1.0 \
    --batch-size 8 --accum-iter 8 --epochs 50 --tta identity,hflip,vflip,hvflip \
    --init-ckpt outputs/outsource_finetune/phase1_aptos/checkpoint-best.pth
```
Then read `outputs/cv/cv_results.json` -> `pooled_oof.macro_sensitivity` and its 95% CI, exactly
as in exp03. A stable win means the pooled lower-CI bound clears 0.765 (and ideally 0.80).
""")

md(r"""
## Notes
- **Domain shift is the risk.** APTOS uses different cameras/populations; that's why we fine-tune
  on local *after* (Phase 2) and select/evaluate only on local. If Phase 2 val macro-sens starts
  *below* the from-scratch run, the APTOS init hurt — lower `CFG_APTOS['epochs']` (less external
  overfit) or reduce Phase-2 `blr` (gentler adaptation).
- **R1 is the true floor** (0.612 pooled): APTOS adds R1 too (~340 imgs), but if R1 stays low,
  an ordinal-aware loss (adjacent-grade penalty) is the complementary next lever.
- APTOS grades are ICDR (severe NPDR=3 folded into R2); a handful of severe cases may look unlike
  local R2 — acceptable noise for a pretraining phase.
""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "retfound", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Outsource-finetune.ipynb")
nbf.write(nb, out)
print("wrote", out, "with", len(cells), "cells")
