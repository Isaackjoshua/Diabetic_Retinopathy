# DATA REPORT — Diabetic Retinopathy / RETFound

## 1. Labels (Reading_Grades.xlsx)
  Using sheet: Sheet2  (canonical one-row-per-patient consensus grade)
  Patients with a grade row: 2401
  Label columns: 'Retinopathy (Left Eye)' -> LE folder ; 'Retinopathy (Right Eye)' -> RE folder
  Per-eye grade distribution (both eyes pooled):
          R0: 2407
          R1: 1360
           U: 621
          R2: 225
         R3A: 188
         R3S: 1

  LATERALITY MAPPING (verified, not assumed):
    * CSV encodes eye by explicit column name 'Left Eye' / 'Right Eye' (no OD/OS on CSV side).
    * Image filenames encode eye by token _LE_/_RE_/_OD_/_OS_.
    * OD (oculus dexter) = RIGHT eye = RE ; OS (oculus sinister) = LEFT eye = LE.
    * Confirmed by patient 2406: RE/ folder contains _OD_ files, LE/ folder contains _OS_ files.
    => LE folder / _LE_ / _OS_  <->  Retinopathy (Left Eye)
    => RE folder / _RE_ / _OD_  <->  Retinopathy (Right Eye)

## 2. Directory structure
  Total image files found: 10484
  Distinct patient folders with images: 2400
  Images NOT directly under an LE/ or RE/ folder (stray subdirs): 57  across 8 patients
  Eye-resolution outcome per image:
                    concordant: 9628
         laterality_discordant: 798
        eye_from_filename_only: 54
                        no_eye: 3
          eye_from_folder_only: 1
  -> 798 images have folder/filename laterality DISAGREEMENT; excluded from labeled set (qc_flag=laterality_discordant).

  Eyes present (patient,eye) with >=1 usable image: 4381
  Images per eye  min/median/max: 1 / 2 / 8

## 3. Reconciliation (folders <-> CSV codes)
  Folders with images: 2400
  CSV patient codes:   2401
  Matched (folder & CSV): 2400
  Folders with NO CSV row (0): []
  CSV codes with NO folder (1): ['1746']

## 4. QC
  Exact byte-identical duplicates (md5): 11 images across 11 groups -> flagged qc=duplicate (kept 1 each). Patient-level split means these cannot leak across splits regardless.
  Corrupt/unreadable images: 0

## 5. Label join
  Images with a resolved eye: 9683
  ...of which joined to a grade row: 9683
  ...ungradable (U) -> unlabeled: 1265
  USABLE labeled images (dedup, non-corrupt, gradable): 8407

  Ordinal class distribution (usable images):
      0               R0_no_dr: 4843
      1                R1_mild: 2730
      2     R2_moderate_severe: 443
      3       R3_proliferative: 391
  Binary referable distribution (usable images):
      0  non_referable: 7573
      1   referable_dr: 834

  Unique labeled EYES: 3834  across 2194 patients
  Ordinal class distribution (unique eyes):
      0               R0_no_dr: 2199
      1                R1_mild: 1253
      2     R2_moderate_severe: 212
      3       R3_proliferative: 170

## 6. Decisions (justified from the data)
  * LABEL SCHEME: 4-class ORDINAL DESP retinopathy grade R0/R1/R2/R3
      (R3A and R3S -> R3; R3S has only 1 eye). Ungradable 'U' eyes are unlabeled/excluded.
      Rationale: DESP grade is ordinal; quadratic-weighted kappa (QWK) is the standard DR
      selection metric. Binary 'referable DR' (R2+) is ALSO stored per image and will be
      reported at eval by collapsing the ordinal head (argmax>=2 / P(R2)+P(R3)).
  * SELECTION METRIC: validation quadratic-weighted Cohen's kappa (QWK). Accuracy is NOT used.
  * SAMPLE SIZE: 8407 usable labeled images / 3834 eyes / 2194 patients.
      Minority classes small (R2 212 eyes, R3 170 eyes).
  * ADAPTATION STRATEGY: FULL fine-tune of ViT-Large with RETFound's recipe
      (layer-wise LR decay 0.65, drop_path, mixup/cutmix, weight decay). Justification:
      ~3834 eyes / ~5.8k train images exceeds RETFound's own downstream benchmarks
      (IDRiD 516, MESSIDOR 1744, APTOS 3662), all of which use full fine-tuning; the CFP
      pretraining makes full fine-tune stable at this N.
  * CLASS IMBALANCE: WeightedRandomSampler over the 4 ordinal classes (compatible with
      RETFound's mixup pipeline). Chosen over class-weighted CE because mixup uses soft targets.
  * SPLIT: patient-level 70/15/15, stratified on each patient's WORST-eye grade (Phase 2).
  * GPU: NVIDIA RTX A4000 16 GB -> batch_size 16 @ 224px, AMP on, grad-accum for eff. batch.

Wrote manifest: /home/eth/Desktop/isaack/Retfound.V2/outputs/manifest.csv  (10484 rows)
