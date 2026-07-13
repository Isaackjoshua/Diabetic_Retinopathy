# Prompt: Fine-tune RETFound for Diabetic Retinopathy Detection (per-eye labels)

You are working in an existing project. Your job is to fine-tune the RETFound retinal
foundation model (https://github.com/rmaphoh/RETFound) to detect diabetic retinopathy (DR)
from color fundus photographs, using the local dataset described below.

Labels are **per eye** (confirmed): each eye has its own DR grade. This resolves the hardest
design question, so you may run end-to-end without a blocking human checkpoint — with ONE
exception, the laterality mapping (see Phase 0). Still work in phases and print a report at each,
but you only need to pause if the eye-mapping convention is ambiguous.

Do not trust the description below — verify it against the actual files in Phase 0.

---

## The data (a hypothesis to test, not ground truth)

- Root directory: `Diabetic Retinopathy IMAGES 2/`
- One folder per patient, named after the patient ID, apparently with a `_T` suffix that must be
  stripped to match the CSV (e.g. folder `ABC123_T` → code `ABC123`).
- Each patient folder contains `LE` (left eye) and `RE` (right eye) subfolders.
- Each `LE`/`RE` subfolder contains one or more color fundus images of that eye.
- A metadata CSV in the same root folder, with a `code` column matching the patient ID
  (folder name minus `_T`), and **per-eye DR labels**.

---

## Non-negotiable constraints

1. **Verify the laterality mapping before joining labels.** Labels are per eye, so the join key is
   `(patient_id, eye)` — and if the eye is mapped wrong, every label is silently inverted. Inspect
   how the CSV encodes eye/laterality (`LE`/`RE`? `left`/`right`? **`OD`/`OS`** where OD=right and
   OS=left? numeric?) and state the exact mapping to the `LE`/`RE` folders. If the convention is
   ambiguous, STOP and ask. Otherwise print the assumed mapping prominently and proceed.
2. **Split at the PATIENT level, never image or eye level.** All images from both eyes of a patient
   go into exactly one of train/val/test. Image- or eye-level splitting leaks patients across
   splits and inflates metrics. Mandatory.
3. **Use color-fundus RETFound weights, not OCT:**
   `--model RETFound_mae --model_arch retfound_mae --finetune RETFound_mae_natureCFP`.
4. **Reuse RETFound's real model + recipe.** Import the backbone, weight loading, positional-embedding
   interpolation, and layer-wise LR decay from the repo (`models_vit.py`, `util/pos_embed.py`,
   `util/lr_decay.py`, `engine_finetune.py`). Do not reimplement them — that's the #1 silent-bug source.
5. **Accuracy is not the primary or selection metric.** See Phase 5.
6. **Pick the adaptation strategy from the actual sample size and justify it.** Don't default to full
   fine-tuning on a small dataset.

---

## Phase 0 — Investigate and report

Write a short exploration script/cells that print a report. Answer with evidence:

**Directory structure**
- Confirm the real layout by walking the root. Deviations from patient → {LE, RE} → images?
  (missing LE or RE, differently-named laterality folders, nested dirs, non-image files.)
- Counts: patients, eyes present, images per eye (min/median/max), total images.
- Inspect files: format, dimensions, colour vs grayscale. Confirm they are color fundus photos,
  not OCT / external-eye photos / scanned reports. Wrong modality = wrong pretrained checkpoint.
- Decode the folder-name → `code` transform exactly (is it always `_T`? any other suffixes?).

**The CSV**
- Print columns, dtypes, first ~10 rows, and value counts of the DR-grade column(s).
- Identify exactly which column(s) hold the per-eye DR label and how eyes are encoded. Resolve the
  laterality mapping to `LE`/`RE` (constraint #1). Flag if it's `OD`/`OS`.
- Determine the label scheme: binary (e.g. referable DR vs not) or multi-class ordinal (ICDR 0–4).
  Report the class distribution at eye level.

**Reconciliation**
- After stripping `_T`: how many folders match a CSV `code`? List folders with no CSV row and CSV
  codes with no folder. How many eyes get a usable label vs unlabeled? Report counts — never drop
  silently.

Print all of this as a **Data Report**. Then proceed to Phase 1 automatically unless the laterality
mapping is ambiguous, in which case stop and ask.

---

## Phase 1 — Image-level manifest

Build one manifest (CSV/Parquet), one row per image:
`image_path, patient_id, eye (LE/RE), dr_label, split(empty), qc_flags`.
The label is joined on `(patient_id, eye)` using the verified laterality mapping. Save it; all later
phases read from it so the mapping stays auditable. Print label distribution by eye and by patient.

Lightweight QC: flag corrupt/unreadable images, obvious non-fundus images, cheap-to-detect duplicates.
Report rows dropped and why.

---

## Phase 2 — Patient-level stratified split

- Split by `patient_id` (GroupShuffleSplit / StratifiedGroupKFold-style). Default 70/15/15 unless
  sample size argues otherwise.
- Stratify on DR grade as well as grouping allows (for ICDR, stratify on the patient's worst-eye grade).
- Assert and print: zero patient overlap across splits; per-split class distributions.
- If any split has near-empty minority classes, say so and recommend k-fold cross-validation instead
  of a single split.
- Persist the split into the manifest with a fixed seed for reproducibility.

---

## Phase 3 — Materialize RETFound's ImageFolder layout

RETFound's `main_finetune.py` expects:

```
data_root/
  train/ <class_name>/ *.jpg
  val/   <class_name>/ *.jpg
  test/  <class_name>/ *.jpg
```

- Generate this from the manifest. **Prefer symlinks over copies**; fall back to copies if symlinks
  aren't supported. Filenames encode `patientid_eye_origname` to preserve provenance and avoid collisions.
- Stable class-folder names mapping cleanly to integer labels (`0..4` for ICDR, or `no_dr`/`referable_dr`).
  Document the class→index mapping and keep it consistent through training and eval.
- Re-verify after building: no patient in >1 split; per-class counts match the manifest.

---

## Phase 4 — The fine-tuning notebook

Deliver a Jupyter notebook. Setup: clone the RETFound repo, install its `requirements.txt`
(PyTorch 2.3+/CUDA 12+). In a markdown cell, note the user must (a) accept the HuggingFace form for
`RETFound_mae_natureCFP` and (b) run `huggingface-cli login --token <TOKEN>` themselves — you can't
complete the gated-access form. Document the `HF_ENDPOINT` mirror fallback if HF is unreachable.

**Model & recipe:** build ViT-Large via `models_vit` with `global_pool`, load `RETFound_mae_natureCFP`,
interpolate positional embeddings with the repo's utility, attach a fresh head sized to the confirmed
class count. Use the repo's layer-wise LR decay and `engine_finetune` train/eval loops. A notebook-native
loop that *imports* these functions is preferred for visibility; if adapting them is brittle, fall back to
`torchrun main_finetune.py` via subprocess with the prepared `data_path`. The recipe must be RETFound's.

**Adaptation strategy (justify from N):**
- Thousands of eyes → full fine-tune is defensible.
- Small/moderate N → linear probing (`--adaptation lp`) or unfreeze only the last few blocks, with
  strong regularization. State the choice and reasoning.

**Config as labeled variables at the top:** `input_size=224`; batch size (default 16; 8 on ~12 GB GPUs
like an RTX 3060; higher on an A6000); epochs; base LR + layer-wise decay; weight decay; warmup; mixed
precision on; class-imbalance handling (weighted loss and/or sampler — pick one, justify from the
distribution). Set and print all seeds. Select the best checkpoint by validation macro-AUROC (or
quadratic weighted kappa for ordinal grades), not accuracy.

---

## Phase 5 — Evaluation

Held-out **test** split only:
- **Ordinal (ICDR 0–4):** quadratic weighted Cohen's kappa (standard DR metric), macro-AUROC (OvR),
  macro-AUPRC, per-class sensitivity/specificity, full confusion matrix. Accuracy secondary.
- **Binary (referable DR):** AUROC, AUPRC, sensitivity & specificity at a stated operating point,
  confusion matrix. On imbalanced data lead with AUPRC and sensitivity.
- Report **eye-level** results (primary). Additionally report **per-patient** results by aggregating
  each patient's two eyes (e.g. worst-eye), showing both.
- Plot ROC/PR curves and the confusion matrix. Save all metrics to JSON/CSV.

Be honest in the summary: tiny test set, near-absent minority class, or unstable metrics → say so and
recommend cross-validation. No single optimistic number presented as settled.

---

## Deliverables
1. Phase 0 Data Report (printed + saved).
2. Image-level manifest with splits.
3. Materialized `train/val/test` ImageFolder (symlinked).
4. Runnable fine-tuning notebook, config surfaced at top, class→index mapping documented.
5. Best checkpoint + metrics report (JSON/CSV) + evaluation plots.

## Do NOT
- Do NOT split at image or eye level — patient-level only.
- Do NOT join labels without verifying the laterality mapping (watch for OD/OS inversion).
- Do NOT use accuracy as primary or selection metric.
- Do NOT reimplement the RETFound backbone / weight loading / pos-embed interpolation / LR-decay — import them.
- Do NOT assume the folder structure or CSV schema; verify in Phase 0.
- Do NOT drop unmatched folders/codes silently — surface them in the reconciliation report.

## Fill in before running
- Exact dataset root path: `/home/eth/Desktop/isaack/Retfound.V2/`
- Exact CSV filename: `Reading_Grades.xlsx`
- Target GPU + VRAM (sets batch size): claude can decide this