"""
Phase 0 (Data Report) + Phase 1 (image-level manifest).

Walks the image tree once, resolves per-image laterality (folder vs filename token,
with concordance QC), joins the per-eye DR grade from Reading_Grades.xlsx Sheet2,
runs lightweight QC (corrupt / duplicate), and writes:
  * outputs/manifest.csv   -- one row per image
  * DATA_REPORT.md         -- the Phase 0 report (also printed to stdout)

Run:  python pipeline/build_manifest.py
"""
import os
import sys
import io
import glob
import hashlib
from collections import Counter, defaultdict

import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

Image.MAX_IMAGE_PIXELS = None  # large fundus images are fine


class Tee:
    """Print to stdout AND capture for saving the report."""
    def __init__(self):
        self.buf = io.StringIO()
    def __call__(self, *args):
        line = " ".join(str(a) for a in args)
        print(line)
        self.buf.write(line + "\n")


def load_labels(say):
    df = pd.read_excel(C.XLSX_PATH, sheet_name=C.XLSX_SHEET)
    df = df[df["code"].notna()].copy()
    df["patient_id"] = df["code"].map(C.strip_code_suffix)
    # one row per patient expected; guard against dupes by keeping first
    dupes = df["patient_id"].duplicated().sum()
    if dupes:
        say(f"  [warn] {dupes} duplicate patient_id rows in {C.XLSX_SHEET}; keeping first")
        df = df.drop_duplicates("patient_id", keep="first")
    # long form: (patient_id, eye) -> grade
    recs = []
    for _, r in df.iterrows():
        recs.append((r["patient_id"], "LE", r[C.LABEL_LEFT_COL]))
        recs.append((r["patient_id"], "RE", r[C.LABEL_RIGHT_COL]))
    lab = pd.DataFrame(recs, columns=["patient_id", "eye", "grade"])
    lab["grade"] = lab["grade"].astype("string").str.strip()
    return df, lab


def walk_images(say):
    rows = []
    exts = (".jpg", ".jpeg", ".png")
    for dirpath, _, files in os.walk(C.IMAGE_ROOT):
        for fn in files:
            if fn.startswith("._"):
                continue  # macOS AppleDouble resource-fork junk, not a real image
            if not fn.lower().endswith(exts):
                say(f"  [non-image] {os.path.join(dirpath, fn)}")
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, C.IMAGE_ROOT)
            pid = C.patient_id_from_path(rel)
            path_eye = C.eye_from_path(rel)
            tok_eye = C.eye_from_filename(fn)
            rows.append({
                "image_path": full,
                "rel_path": rel,
                "patient_id": pid,
                "path_eye": path_eye,
                "token_eye": tok_eye,
                "acq_id": C.acquisition_id(fn),
                "filename": fn,
            })
    return pd.DataFrame(rows)


def resolve_eye(row):
    """Return (eye, source_flag). Concordance-first.
    NB: use pd.notna, not truthiness -- a NaN path_eye is truthy and would be
    misread as 'present' by `if p and t`."""
    p = row["path_eye"] if pd.notna(row["path_eye"]) else None
    t = row["token_eye"] if pd.notna(row["token_eye"]) else None
    if p and t:
        if p == t:
            return p, "concordant"
        return None, "laterality_discordant"   # exclude, ambiguous eye
    if t and not p:
        return t, "eye_from_filename_only"      # stray dir; token is only signal
    if p and not t:
        return p, "eye_from_folder_only"
    return None, "no_eye"


def main():
    os.makedirs(C.OUT_DIR, exist_ok=True)
    say = Tee()
    say("# DATA REPORT — Diabetic Retinopathy / RETFound")
    say("")

    # ---------------- CSV ----------------
    say("## 1. Labels (Reading_Grades.xlsx)")
    labdf, lab = load_labels(say)
    say(f"  Using sheet: {C.XLSX_SHEET}  (canonical one-row-per-patient consensus grade)")
    say(f"  Patients with a grade row: {labdf['patient_id'].nunique()}")
    say(f"  Label columns: '{C.LABEL_LEFT_COL}' -> LE folder ; '{C.LABEL_RIGHT_COL}' -> RE folder")
    say("  Per-eye grade distribution (both eyes pooled):")
    for g, n in lab["grade"].value_counts(dropna=False).items():
        say(f"      {g!s:>6}: {n}")
    say("")
    say("  LATERALITY MAPPING (verified, not assumed):")
    say("    * CSV encodes eye by explicit column name 'Left Eye' / 'Right Eye' (no OD/OS on CSV side).")
    say("    * Image filenames encode eye by token _LE_/_RE_/_OD_/_OS_.")
    say("    * OD (oculus dexter) = RIGHT eye = RE ; OS (oculus sinister) = LEFT eye = LE.")
    say("    * Confirmed by patient 2406: RE/ folder contains _OD_ files, LE/ folder contains _OS_ files.")
    say("    => LE folder / _LE_ / _OS_  <->  Retinopathy (Left Eye)")
    say("    => RE folder / _RE_ / _OD_  <->  Retinopathy (Right Eye)")
    say("")

    # ---------------- Directory walk ----------------
    say("## 2. Directory structure")
    img = walk_images(say)
    say(f"  Total image files found: {len(img)}")
    say(f"  Distinct patient folders with images: {img['patient_id'].nunique()}")
    # deviations
    stray = img[img["path_eye"].isna()]
    say(f"  Images NOT directly under an LE/ or RE/ folder (stray subdirs): {len(stray)}"
        f"  across {stray['patient_id'].nunique()} patients")
    # eye resolution
    res = img.apply(resolve_eye, axis=1, result_type="expand")
    img["eye"] = res[0]
    img["eye_source"] = res[1]
    say("  Eye-resolution outcome per image:")
    for k, n in img["eye_source"].value_counts().items():
        say(f"      {k:>24}: {n}")
    disc = (img["eye_source"] == "laterality_discordant").sum()
    say(f"  -> {disc} images have folder/filename laterality DISAGREEMENT; "
        f"excluded from labeled set (qc_flag=laterality_discordant).")
    say("")

    # images per eye stats
    per_eye = img[img["eye"].notna()].groupby(["patient_id", "eye"]).size()
    say(f"  Eyes present (patient,eye) with >=1 usable image: {len(per_eye)}")
    say(f"  Images per eye  min/median/max: {per_eye.min()} / {int(per_eye.median())} / {per_eye.max()}")
    say("")

    # ---------------- Reconciliation ----------------
    say("## 3. Reconciliation (folders <-> CSV codes)")
    folder_ids = set(img["patient_id"].unique())
    csv_ids = set(labdf["patient_id"].unique())
    say(f"  Folders with images: {len(folder_ids)}")
    say(f"  CSV patient codes:   {len(csv_ids)}")
    say(f"  Matched (folder & CSV): {len(folder_ids & csv_ids)}")
    only_folder = sorted(folder_ids - csv_ids)
    only_csv = sorted(csv_ids - folder_ids)
    say(f"  Folders with NO CSV row ({len(only_folder)}): {only_folder[:20]}{' ...' if len(only_folder)>20 else ''}")
    say(f"  CSV codes with NO folder ({len(only_csv)}): {only_csv[:20]}{' ...' if len(only_csv)>20 else ''}")
    say("")

    # ---------------- QC: duplicates ----------------
    # EXACT byte-duplicates only (md5). NOTE: files sharing an acquisition-id with
    # ' (1)'/' (2)' suffixes were checked and are DISTINCT frames of the same eye
    # (different focus/exposure), NOT duplicates -- so they are kept as legit multi-view.
    say("## 4. QC")
    def md5(path, chunk=1 << 20):
        h = hashlib.md5()
        try:
            with open(path, "rb") as fh:
                for b in iter(lambda: fh.read(chunk), b""):
                    h.update(b)
        except Exception:
            return None
        return h.hexdigest()
    img["md5"] = img["image_path"].map(md5)
    img["qc_dupe"] = False
    dup_groups = 0
    for hv, grp in img[img["md5"].notna()].groupby("md5"):
        if len(grp) > 1:
            dup_groups += 1
            img.loc[grp.index[1:], "qc_dupe"] = True  # keep first, flag rest
    n_dupe = int(img["qc_dupe"].sum())
    say(f"  Exact byte-identical duplicates (md5): {n_dupe} images across {dup_groups} "
        f"groups -> flagged qc=duplicate (kept 1 each). Patient-level split means these "
        f"cannot leak across splits regardless.")

    # QC: corrupt check (verify headers; cheap). Full decode is slow on 10k imgs -> verify() only.
    corrupt = []
    for path in img["image_path"]:
        try:
            with Image.open(path) as im:
                im.verify()
        except Exception as e:  # noqa
            corrupt.append(path)
    img["qc_corrupt"] = img["image_path"].isin(set(corrupt))
    say(f"  Corrupt/unreadable images: {len(corrupt)}")
    say("")

    # ---------------- Join labels ----------------
    say("## 5. Label join")
    m = img.merge(lab, on=["patient_id", "eye"], how="left")
    m["grade"] = m["grade"].astype("string")
    m["dr_label"] = m["grade"].map(C.GRADE_TO_ORDINAL)          # 0..3 or NaN
    m["referable"] = m["grade"].map(C.GRADE_TO_BINARY)          # 0/1 or NaN
    # qc flags string
    def qc_flags(r):
        f = []
        if r["eye_source"] in ("laterality_discordant", "no_eye"):
            f.append(r["eye_source"])
        if r["eye_source"] in ("eye_from_filename_only", "eye_from_folder_only"):
            f.append(r["eye_source"])
        if r["qc_dupe"]:
            f.append("duplicate")
        if r["qc_corrupt"]:
            f.append("corrupt")
        if pd.isna(r["grade"]):
            f.append("no_label")
        elif r["grade"] in C.UNGRADABLE:
            f.append("ungradable")
        return ";".join(f)
    m["qc_flags"] = m.apply(qc_flags, axis=1)

    # usable = has eye, has ordinal label, not corrupt, not dupe
    m["usable"] = (
        m["eye"].notna()
        & m["dr_label"].notna()
        & (~m["qc_corrupt"])
        & (~m["qc_dupe"])
    )
    say(f"  Images with a resolved eye: {int(m['eye'].notna().sum())}")
    say(f"  ...of which joined to a grade row: {int(m['grade'].notna().sum())}")
    say(f"  ...ungradable (U) -> unlabeled: {int((m['grade'].isin(C.UNGRADABLE)).sum())}")
    say(f"  USABLE labeled images (dedup, non-corrupt, gradable): {int(m['usable'].sum())}")
    say("")
    say("  Ordinal class distribution (usable images):")
    u = m[m["usable"]]
    for k in sorted(C.ORDINAL_CLASS_NAMES):
        say(f"      {k} {C.ORDINAL_CLASS_NAMES[k]:>22}: {int((u['dr_label']==k).sum())}")
    say("  Binary referable distribution (usable images):")
    for k in sorted(C.BINARY_CLASS_NAMES):
        say(f"      {k} {C.BINARY_CLASS_NAMES[k]:>14}: {int((u['referable']==k).sum())}")
    say("")
    # per-eye (unique eyes) distribution
    eyes = u.drop_duplicates(["patient_id", "eye"])
    say(f"  Unique labeled EYES: {len(eyes)}  across {eyes['patient_id'].nunique()} patients")
    say("  Ordinal class distribution (unique eyes):")
    for k in sorted(C.ORDINAL_CLASS_NAMES):
        say(f"      {k} {C.ORDINAL_CLASS_NAMES[k]:>22}: {int((eyes['dr_label']==k).sum())}")
    say("")

    # ---------------- Decisions ----------------
    n_img = int(u['usable'].sum()) if 'usable' in u else int(m['usable'].sum())
    n_eyes = len(eyes); n_pat = eyes['patient_id'].nunique()
    say("## 6. Decisions (justified from the data)")
    say(f"  * LABEL SCHEME: 4-class ORDINAL DESP retinopathy grade R0/R1/R2/R3")
    say(f"      (R3A and R3S -> R3; R3S has only 1 eye). Ungradable 'U' eyes are unlabeled/excluded.")
    say(f"      Rationale: DESP grade is ordinal; quadratic-weighted kappa (QWK) is the standard DR")
    say(f"      selection metric. Binary 'referable DR' (R2+) is ALSO stored per image and will be")
    say(f"      reported at eval by collapsing the ordinal head (argmax>=2 / P(R2)+P(R3)).")
    say(f"  * SELECTION METRIC: validation quadratic-weighted Cohen's kappa (QWK). Accuracy is NOT used.")
    say(f"  * SAMPLE SIZE: {n_img} usable labeled images / {n_eyes} eyes / {n_pat} patients.")
    say(f"      Minority classes small (R2 {int((eyes['dr_label']==2).sum())} eyes, "
        f"R3 {int((eyes['dr_label']==3).sum())} eyes).")
    say(f"  * ADAPTATION STRATEGY: FULL fine-tune of ViT-Large with RETFound's recipe")
    say(f"      (layer-wise LR decay 0.65, drop_path, mixup/cutmix, weight decay). Justification:")
    say(f"      ~{n_eyes} eyes / ~5.8k train images exceeds RETFound's own downstream benchmarks")
    say(f"      (IDRiD 516, MESSIDOR 1744, APTOS 3662), all of which use full fine-tuning; the CFP")
    say(f"      pretraining makes full fine-tune stable at this N.")
    say(f"  * CLASS IMBALANCE: WeightedRandomSampler over the 4 ordinal classes (compatible with")
    say(f"      RETFound's mixup pipeline). Chosen over class-weighted CE because mixup uses soft targets.")
    say(f"  * SPLIT: patient-level 70/15/15, stratified on each patient's WORST-eye grade (Phase 2).")
    say(f"  * GPU: NVIDIA RTX A4000 16 GB -> batch_size 16 @ 224px, AMP on, grad-accum for eff. batch.")
    say("")

    # ---------------- Save manifest ----------------
    keep = ["image_path", "patient_id", "eye", "grade", "dr_label", "referable",
            "eye_source", "qc_flags", "usable", "filename"]
    manifest = m[keep].copy()
    manifest["split"] = ""   # filled in Phase 2
    manifest.to_csv(C.MANIFEST_PATH, index=False)
    say(f"Wrote manifest: {C.MANIFEST_PATH}  ({len(manifest)} rows)")

    with open(C.DATA_REPORT_PATH, "w") as f:
        f.write(say.buf.getvalue())
    say(f"Wrote data report: {C.DATA_REPORT_PATH}")


if __name__ == "__main__":
    main()
