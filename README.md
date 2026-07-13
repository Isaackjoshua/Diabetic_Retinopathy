# RETFound → Diabetic Retinopathy (per-eye) fine-tuning

Fine-tunes the **RETFound colour-fundus** foundation model (`RETFound_mae_natureCFP`,
ViT-Large) to grade diabetic retinopathy on a local screening dataset with **per-eye** labels.

## TL;DR — verified facts (details in `DATA_REPORT.md`)
- **Laterality (the label-inverting trap):** CSV Sheet2 uses explicit `Retinopathy (Left Eye)`→LE,
  `Retinopathy (Right Eye)`→RE. Filenames use `_OD_`/`_OS_` = **OD(right)=RE, OS(left)=LE**,
  confirmed by patient `2406`. 798 images whose folder & filename disagree are **excluded**, not relabeled.
- **Join:** folder id (e.g. `0019`) = CSV `code` with the trailing `_T` stripped (`0019_T`).
- **Label scheme:** 4-class ordinal NHS-DESP **R0/R1/R2/R3** (R3A+R3S→R3; `U`=ungradable excluded).
  Binary **referable DR (R2+)** is derived from the same head at eval.
- **Usable data:** 8407 images / 3834 eyes / 2194 patients. Patient-level 70/15/15 split.
- **Selection metric:** validation **quadratic-weighted kappa (QWK)** — never accuracy.
- **Adaptation:** full fine-tune, RETFound recipe (layer-wise LR decay 0.65, weighted-CE for imbalance).
- **GPU:** RTX A4000 16 GB → batch 16 @224 (peaks ~7 GB), effective batch 64.

## Pipeline (reproducible)
```bash
conda activate retfound
python pipeline/build_manifest.py          # Phase 0 report + Phase 1 manifest  -> DATA_REPORT.md, outputs/manifest.csv
python pipeline/make_split.py              # Phase 2 patient-level stratified split (writes split into manifest)
python pipeline/materialize_imagefolder.py # Phase 3 symlinked ImageFolder -> outputs/dr_imagefolder/, class_mapping.json
python pipeline/build_resized_cache.py     # speed cache: short-side 512 copies -> outputs/dr_imagefolder_cache/ (~30x faster decode)
# Phase 4/5: train (needs gated HF weights + login)
jupyter lab RETFound_DR_finetune.ipynb
# Evaluation: full metrics report on the fine-tuned checkpoint (no HF access needed)
jupyter lab evaluation/DR_evaluation.ipynb
```

## Files
| Path | What |
|------|------|
| `DATA_REPORT.md` | Phase 0 audit (structure, CSV, laterality proof, reconciliation, decisions) |
| `outputs/manifest.csv` | one row per image: path, patient_id, eye, grade, dr_label, referable, eye_source, qc_flags, usable, split |
| `outputs/dr_imagefolder/` | `train|val|test/<class>/` symlinks; names `pid_eye_orig` preserve provenance |
| `outputs/dr_imagefolder_cache/` | same layout, images resized to short-side 512 (training speed cache; notebook's default `data_path`) |
| `outputs/class_mapping.json` | `R0_no_dr=0, R1_mild=1, R2_moderate_severe=2, R3_proliferative=3` |
| `RETFound_DR_finetune.ipynb` | baseline training + evaluation notebook (224px, weighted-CE, QWK selection) |
| `experiment01.ipynb` | Experiment 01 training: 384px + focal loss (γ=2) + checkpoint selection by macro-sensitivity → `outputs/experiment01/` |
| `evaluation/DR_evaluation.ipynb` | standalone eval of the **baseline** checkpoint |
| `evaluation/experiment01_evaluation.ipynb` | standalone eval of the **experiment 01** checkpoint |
| `pipeline/dr_losses.py` | `FocalLoss` (multi-class, α-weighted) used by experiment 01 |

Evaluation notebooks report precision/recall/sensitivity/specificity/F1/AUROC/AUPRC/QWK, confusion matrices, ROC+PR curves, per-class bars, and a referable-DR operating-point sweep (→ each notebook's `evaluation/results*/`).
| `pipeline/common.py` | shared constants + laterality/label helpers |
| `pipeline/dr_train.py` | wrappers that **import** RETFound's real model / lr_decay / datasets / engine |
| `pipeline/dr_eval.py` | Phase 5 metrics + eye/patient aggregation + ROC/PR/confusion plots |

## Before running the notebook (manual, one-time)
1. Accept the gated form: https://huggingface.co/YukunZhou/RETFound_mae_natureCFP
2. `huggingface-cli login --token <YOUR_HF_TOKEN>`
3. (If HF blocked) `export HF_ENDPOINT=https://hf-mirror.com`

A subagent/assistant **cannot** complete the gated form, so the final trained checkpoint
and real metrics are produced when *you* run the notebook after step 1–2. Every other
component (data build, model architecture, training loop, evaluation, plotting) has been
smoke-tested end-to-end; only the gated weight download is left to you.

## Honest caveats
Minority classes are small at eye level (test ≈ 35 R2, 22 R3). Treat single-number QWK /
per-class sensitivity / referable-DR PR as indicative; use k-fold CV before any deployment claim.
Lead with AUPRC + sensitivity for referable DR (prevalence ≈ 10 %).
