"""Materialize the APTOS-2019 dataset as a class-consistent ImageFolder for external
pre-fine-tuning (option #1: transfer from a large public DR set to enrich rare grades).

APTOS labels are ICDR 0-4; we map them onto the SAME 4-class NHS ordinal scheme + class
indices used locally (common.ORDINAL_CLASS_NAMES), so a model's 4-way head transfers directly
between the APTOS phase and the local phase:

    ICDR 0 (no DR)      -> R0  (0)
    ICDR 1 (mild)       -> R1  (1)
    ICDR 2 (moderate)   -> R2  (2)   # R2 = moderate/severe NPDR
    ICDR 3 (severe)     -> R2  (2)
    ICDR 4 (PDR)        -> R3  (3)   # proliferative

Output: outputs/aptos_imagefolder/{train,val,test}/<class_name>/<id>.png  (symlinks).
Idempotent: rebuilds the tree each run. Only rows whose image file exists on disk are kept.
"""
import os
import sys
import glob
import shutil
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APTOS_DIR = os.path.join(PROJECT_ROOT, "APTOS-2019")
OUT_DIR = os.path.join(C.OUT_DIR, "aptos_imagefolder")

ICDR_TO_ORDINAL = {0: 0, 1: 1, 2: 2, 3: 2, 4: 3}

# (csv, image subdir, output split name)
SPLITS = [
    ("train_1.csv", os.path.join("train_images", "train_images"), "train"),
    ("valid.csv",   os.path.join("val_images", "val_images"),     "val"),
    ("test.csv",    os.path.join("test_images", "test_images"),   "test"),
]


def _resolve_image(img_dir, id_code):
    """Find the image file for an id_code (png confirmed, but glob any extension)."""
    exact = os.path.join(img_dir, f"{id_code}.png")
    if os.path.exists(exact):
        return exact
    hits = glob.glob(os.path.join(img_dir, f"{id_code}.*"))
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aptos-dir", default=APTOS_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    assert os.path.isdir(args.aptos_dir), f"APTOS dir not found: {args.aptos_dir}"
    if os.path.isdir(args.out_dir):
        shutil.rmtree(args.out_dir)

    grand = {}
    for csv_name, img_rel, split in SPLITS:
        csv_path = os.path.join(args.aptos_dir, csv_name)
        img_dir = os.path.join(args.aptos_dir, img_rel)
        if not os.path.exists(csv_path):
            print(f"  [skip] {csv_name} missing"); continue
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        assert {"id_code", "diagnosis"} <= set(df.columns), f"unexpected columns in {csv_name}: {df.columns}"
        # drop rows with no/blank diagnosis (some APTOS test rows are unlabeled)
        df = df[pd.to_numeric(df["diagnosis"], errors="coerce").notna()].copy()
        df["diagnosis"] = df["diagnosis"].astype(int)

        kept, missing, counts = 0, 0, {}
        for _, r in df.iterrows():
            if int(r["diagnosis"]) not in ICDR_TO_ORDINAL:
                continue
            src = _resolve_image(img_dir, str(r["id_code"]).strip())
            if src is None:
                missing += 1; continue
            cls_idx = ICDR_TO_ORDINAL[int(r["diagnosis"])]
            cls_name = C.ORDINAL_CLASS_NAMES[cls_idx]
            dst_dir = os.path.join(args.out_dir, split, cls_name)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, f"{str(r['id_code']).strip()}.png")
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(src), dst)
            kept += 1
            counts[cls_name] = counts.get(cls_name, 0) + 1
        grand[split] = counts
        print(f"[{split}] kept {kept} images (missing files: {missing}) | {dict(sorted(counts.items()))}")

    # summary table (ordinal order)
    print("\n=== APTOS ImageFolder (mapped to R0-R3) ===")
    names = [C.ORDINAL_CLASS_NAMES[i] for i in range(4)]
    header = f"{'split':>6} | " + " | ".join(f"{n:>20}" for n in names) + " |   total"
    print(header)
    for split in ("train", "val", "test"):
        c = grand.get(split, {})
        row = f"{split:>6} | " + " | ".join(f"{c.get(n,0):>20}" for n in names) + f" | {sum(c.values()):>6}"
        print(row)
    print(f"\nwrote -> {args.out_dir}")


if __name__ == "__main__":
    main()
