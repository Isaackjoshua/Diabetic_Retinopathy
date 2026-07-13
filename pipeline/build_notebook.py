"""Generate RETFound_DR_finetune.ipynb (Phase 4 + Phase 5)."""
import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# RETFound → Diabetic Retinopathy fine-tuning (per-eye labels)

Fine-tunes the **RETFound colour-fundus** foundation model (`RETFound_mae_natureCFP`,
ViT-Large) to grade diabetic retinopathy on a local Tanzanian screening dataset.

This notebook is the training+evaluation front-end. Phases 0–3 (data audit, manifest,
patient-level split, ImageFolder) were produced by the scripts in `pipeline/` and are
summarised in **`DATA_REPORT.md`**. Re-run them with the first code cell if needed.

### Verified laterality mapping (the one thing that silently inverts every label)
- CSV (`Reading_Grades.xlsx`, **Sheet2**, one row/patient) labels each eye by explicit
  column name: **`Retinopathy (Left Eye)` → LE**, **`Retinopathy (Right Eye)` → RE**.
- Image filenames encode the eye by token `_LE_/_RE_/_OD_/_OS_`, with the standard
  ophthalmic convention **OD (dexter)=RIGHT=RE, OS (sinister)=LEFT=LE**, independently
  confirmed by patient `2406` (its `RE/` folder holds `_OD_` files, `LE/` holds `_OS_`).
- Images whose folder and filename disagree on laterality (798) are **excluded** (ambiguous eye).

### Task & label scheme (justified in DATA_REPORT.md §6)
- **4-class ordinal** NHS-DESP grade **R0/R1/R2/R3** (R3A+R3S→R3; `U`=ungradable excluded).
- Class→index: `R0_no_dr=0, R1_mild=1, R2_moderate_severe=2, R3_proliferative=3`.
- **Selection metric = validation quadratic-weighted Cohen's kappa (QWK)** — *not accuracy*.
- The binary **referable-DR (R2+)** view is derived from the same head at evaluation.

Usable data: **8407 images / 3834 eyes / 2194 patients**; patient-level 70/15/15 split.
""")

md(r"""
## ⚠️ Gated model access — do this once before running

`RETFound_mae_natureCFP` is a **gated** Hugging Face model. You must:

1. Open <https://huggingface.co/YukunZhou/RETFound_mae_natureCFP> and **accept the access form**.
2. Authenticate in this environment (token from <https://huggingface.co/settings/tokens>):
   ```bash
   huggingface-cli login --token <YOUR_HF_TOKEN>
   ```
   *(A subagent/assistant cannot complete the gated form for you — this step is manual.)*
3. If Hugging Face is unreachable from your network, set a mirror **before** the import cell:
   ```python
   import os; os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
   ```

The download is triggered in the **“Build model”** cell (`hf_hub_download`). A 403
`GatedRepoError` there means step 1 or 2 is incomplete.
""")

code(r"""
# --- (optional) rebuild Phases 0-3 from scratch. Skips if the ImageFolder already exists. ---
import os, subprocess, sys
PROJECT = os.path.abspath(".")
assert os.path.isdir("pipeline"), "Run this notebook from the project root (Retfound.V2/)."
if not os.path.isdir("outputs/dr_imagefolder"):
    for script in ["build_manifest.py", "make_split.py", "materialize_imagefolder.py"]:
        print("running", script); subprocess.run([sys.executable, f"pipeline/{script}"], check=True)
# speed cache: resize the 12MP JPEGs to shorter-side 512 once (~30x faster decode per epoch)
if not os.path.isdir("outputs/dr_imagefolder_cache"):
    print("building resize speed-cache (one-time, a few minutes)...")
    subprocess.run([sys.executable, "pipeline/build_resized_cache.py", "--size", "512"], check=True)
# clone RETFound if missing + install its requirements
if not os.path.isdir("RETFound_repo"):
    subprocess.run(["git","clone","--depth","1","https://github.com/rmaphoh/RETFound.git","RETFound_repo"], check=True)
    subprocess.run([sys.executable,"-m","pip","install","-q","-r","RETFound_repo/requirements.txt"], check=True)
print("ready")
""")

code(r"""
# ============================ CONFIG (all knobs here) ============================
CONFIG = dict(
    # data / model
    data_path   = "outputs/dr_imagefolder_cache",  # resized (short-side 512) speed cache;
                                               # use "outputs/dr_imagefolder" for full-res symlinks

    nb_classes  = 4,                            # R0,R1,R2,R3 ordinal
    input_size  = 224,                          # RETFound CFP native
    finetune_id = "RETFound_mae_natureCFP",     # GATED colour-fundus weights (NOT OCT)
    drop_path   = 0.2,

    # adaptation strategy: FULL fine-tune (justified: ~3.8k eyes / ~5.8k train imgs exceeds
    # RETFound's own downstream benchmarks IDRiD/MESSIDOR/APTOS, all full fine-tuned).
    adaptation  = "finetune",                   # "finetune" (full) or "lp" (linear probe)

    # optimisation (RETFound recipe)
    batch_size    = 16,     # A4000 16GB: peaks ~7GB @224; raise on bigger GPUs
    accum_iter    = 4,      # effective batch = 16*4 = 64
    epochs        = 50,
    warmup_epochs = 10,
    blr           = 5e-3,   # lr = blr * eff_batch/256  -> 1.25e-3
    layer_decay   = 0.65,   # layer-wise LR decay (imported from util.lr_decay)
    weight_decay  = 0.05,
    min_lr        = 1e-6,
    clip_grad     = None,

    # class-imbalance: inverse-frequency WEIGHTED cross-entropy (mixup is OFF in this
    # recipe, so soft-target issues don't arise). See class weights printed below.
    # (Alternative WeightedRandomSampler noted in DATA_REPORT.md; weighted-CE chosen for simplicity.)

    # runtime
    device      = "cuda",
    seed        = 42,
    num_workers = 10,
    output_dir  = "outputs/train_run",
    task        = "dr_retfound_r0r3",
)
SELECTION_METRIC = "qwk"   # validation quadratic-weighted kappa -> best checkpoint
import os; os.makedirs(CONFIG["output_dir"], exist_ok=True)
CONFIG
""")

code(r"""
# ============================ imports, seeds, device ============================
import os, sys, json, time, copy
import numpy as np, torch
sys.path.insert(0, "pipeline"); sys.path.insert(0, "RETFound_repo")

import dr_train as T          # thin wrappers that IMPORT RETFound's real code
import dr_eval as E           # Phase-5 metrics / aggregation
from engine_finetune import train_one_epoch          # RETFound's train loop (imported, not reimplemented)

args = T.make_args(CONFIG)
T.set_seed(CONFIG["seed"])
torch.backends.cudnn.benchmark = True
device = torch.device(CONFIG["device"] if torch.cuda.is_available() else "cpu")
print("device:", device, "| torch", torch.__version__)
""")

code(r"""
# ============================ data + class weights ============================
(ds_tr, ds_va, ds_te), (dl_tr, dl_va, dl_te) = T.build_loaders(args, shuffle_train=True)
print("images  train/val/test:", len(ds_tr), len(ds_va), len(ds_te))
print("class_to_idx:", ds_tr.class_to_idx)

# sanity: the ImageFolder mapping must equal our documented class->index
import json as _json
CM = _json.load(open("outputs/class_mapping.json"))
assert ds_tr.class_to_idx == CM["ordinal_class_to_index"], "class mapping mismatch!"

class_weights, counts = T.class_weights_from_dataset(ds_tr, CONFIG["nb_classes"], device)
print("train class counts :", counts)
print("CE class weights   :", class_weights.cpu().numpy().round(3))
""")

code(r"""
# ============================ build model + load GATED weights ============================
model = T.build_model_arch(args)                      # ViT-L via models_vit (global_pool)
msg = T.load_pretrained(model, args)                  # hf_hub_download + interpolate_pos_embed + strict=False
model.to(device)
# expected missing = the freshly-initialised classifier head + global-pool norm
# (head.weight/bias, fc_norm.weight/bias). Unexpected = MAE decoder keys, correctly dropped.
print("missing keys (expect head.* + fc_norm.*):", list(msg.missing_keys))
print(f"unexpected keys: {len(msg.unexpected_keys)} (MAE decoder / replaced encoder norm — discarded)")
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable params: {n_train/1e6:.1f} M  | adaptation={args.adaptation}")
""")

code(r"""
# ============================ optimizer / criterion / scaler ============================
optimizer, loss_scaler = T.build_optimizer(model, args)    # param_groups_lrd + AdamW + AMP scaler
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
print(f"param groups: {len(optimizer.param_groups)} | base lr: {args.lr:.2e} | eff batch: {args.batch_size*args.accum_iter}")
print("criterion:", criterion)
""")

code(r"""
# ============================ training loop (select best by val QWK) ============================
from sklearn.metrics import cohen_kappa_score, roc_auc_score

def val_scores():
    y, p = E.predict(model, dl_va, device)
    qwk = cohen_kappa_score(y, p.argmax(1), weights="quadratic", labels=list(range(CONFIG["nb_classes"])))
    try:
        yoh = np.eye(CONFIG["nb_classes"])[y]; cols = [c for c in range(CONFIG["nb_classes"]) if yoh[:,c].sum()>0]
        auroc = roc_auc_score(yoh[:,cols], p[:,cols], average="macro", multi_class="ovr")
    except Exception: auroc = float("nan")
    return float(qwk), float(auroc)

best_score, best_epoch, best_state, history = -1.0, -1, None, []
ckpt_path = os.path.join(CONFIG["output_dir"], "checkpoint-best.pth")
t0 = time.time()
for epoch in range(CONFIG["epochs"]):
    tr = train_one_epoch(model, criterion, dl_tr, optimizer, device, epoch,
                         loss_scaler, args.clip_grad, None, None, args)
    qwk, auroc = val_scores()
    score = qwk if SELECTION_METRIC == "qwk" else auroc
    history.append({"epoch": epoch, "train_loss": tr["loss"], "val_qwk": qwk, "val_macro_auroc": auroc})
    tag = ""
    if score > best_score:
        best_score, best_epoch = score, epoch
        best_state = copy.deepcopy(model.state_dict())
        torch.save({"model": best_state, "epoch": epoch, "config": CONFIG,
                    "val_qwk": qwk, "val_macro_auroc": auroc}, ckpt_path)
        tag = "  <-- best"
    print(f"epoch {epoch:02d}  train_loss={tr['loss']:.4f}  val_QWK={qwk:.4f}  val_AUROC={auroc:.4f}{tag}")
json.dump(history, open(os.path.join(CONFIG["output_dir"], "history.json"), "w"), indent=2)
print(f"\nDone in {(time.time()-t0)/60:.1f} min. Best epoch {best_epoch}  {SELECTION_METRIC}={best_score:.4f}")
""")

code(r"""
# ---- training curves ----
import matplotlib.pyplot as plt
h = history
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot([x["epoch"] for x in h], [x["train_loss"] for x in h]); ax[0].set_title("train loss"); ax[0].set_xlabel("epoch")
ax[1].plot([x["epoch"] for x in h], [x["val_qwk"] for x in h], label="val QWK")
ax[1].plot([x["epoch"] for x in h], [x["val_macro_auroc"] for x in h], label="val macro-AUROC")
ax[1].axvline(best_epoch, ls="--", c="grey"); ax[1].legend(); ax[1].set_title("validation"); ax[1].set_xlabel("epoch")
fig.tight_layout(); plt.show()
""")

md(r"""
## Phase 5 — Evaluation on the held-out **test** split

Loads the best checkpoint (selected by validation QWK) and reports, at **image**,
**eye** (primary), and **patient** (worst-eye) level:
- Ordinal: **QWK**, macro-AUROC (OvR), macro-AUPRC, per-class sensitivity/specificity, confusion matrix.
- Binary **referable DR (R2+)**: AUROC, AUPRC, sensitivity & specificity at a stated operating point.

Metrics saved to `outputs/train_run/eval_test/metrics.json`; ROC/PR + confusion plots saved alongside.
""")

code(r"""
# ============================ evaluate on TEST ============================
best = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(best["model"]); model.to(device)
print(f"Loaded best checkpoint: epoch {best['epoch']}  val_QWK={best.get('val_qwk'):.4f}")

test_paths = [p for p, _ in ds_te.samples]
y_true, y_prob = E.predict(model, dl_te, device)
eval_dir = os.path.join(CONFIG["output_dir"], "eval_test")
report = E.full_report(test_paths, y_true, y_prob, eval_dir)

def show(level):
    r = report[level]; b = r["binary_referable"]
    print(f"\n===== {level.upper()} (n={r['n']}) =====")
    print(f"  QWK={r['qwk']:.4f}  macroAUROC={r['macro_auroc_ovr']:.4f}  macroAUPRC={r['macro_auprc']:.4f}  "
          f"macroF1={r['macro_f1']:.4f}  acc={r['accuracy']:.4f}")
    print("  per-class sens/spec:", {k: (round(v['sensitivity'],3), round(v['specificity'],3), v['support'])
                                     for k, v in r['per_class'].items()})
    if "auroc" in b:
        print(f"  referable-DR: AUROC={b['auroc']:.4f}  AUPRC={b['auprc']:.4f}  "
              f"sens={b['sensitivity']:.3f}  spec={b['specificity']:.3f} @op={b['operating_point']}  (pos={b['n_pos']})")
for lv in ["image_level", "eye_level", "patient_level"]:
    show(lv)
print("\nsaved:", eval_dir)
""")

code(r"""
# ---- show saved plots ----
from IPython.display import Image, display
for f in ["roc_pr_eye.png", "confusion_eye.png", "roc_pr_patient.png", "confusion_patient.png"]:
    p = os.path.join(eval_dir, f)
    if os.path.exists(p): display(Image(p))
""")

md(r"""
## Honest reporting notes
- **Minority classes are small** at eye level (test ≈ 35 R2, 22 R3 eyes). QWK / per-class
  sensitivity for R2–R3 and the referable-DR PR curve are estimated on few positives —
  treat single numbers as indicative, not settled. Prefer the ranges you'd get from
  **k-fold CV** if a firm estimate is needed (recommended before any deployment claim).
- **Lead with AUPRC + sensitivity** for referable DR (prevalence ≈ 10 %); accuracy is
  reported only as a secondary descriptor.
- **798 laterality-discordant images were excluded**, not silently relabeled; trusting the
  filename token instead would recover them but risks inverted per-eye labels.
- The model selection metric is **validation QWK**, never accuracy.
""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "retfound", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "RETFound_DR_finetune.ipynb")
nbf.write(nb, out)
print("wrote", out, "with", len(cells), "cells")
