"""
Phase 2 -- patient-level stratified split (70/15/15).

Splits by patient_id, stratified on each patient's WORST-eye ordinal grade.
Because we stratify a patient-level table (one row per patient), a stratified
train/test split of patients is inherently group-safe: no patient can appear in
two splits. Writes the split back into outputs/manifest.csv.

Run:  python pipeline/make_split.py
"""
import os
import sys
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

SEED = 42
TEST_FRAC = 0.15
VAL_FRAC = 0.15  # of the whole


def add_kfolds(m, u, kfolds):
    """Assign each patient to a CV fold (patient-level, stratified on worst-eye grade)
    and write a `fold` column (0..K-1) into the manifest. Group-safe by construction."""
    pat = u.groupby("patient_id")["dr_label"].max().reset_index()
    pat.columns = ["patient_id", "worst_grade"]
    skf = StratifiedKFold(n_splits=kfolds, shuffle=True, random_state=SEED)
    fold_of = {}
    for f, (_, te_idx) in enumerate(skf.split(pat["patient_id"], pat["worst_grade"])):
        for pid in pat.iloc[te_idx]["patient_id"]:
            fold_of[pid] = f
    # string dtype so non-usable rows can hold "" like the `split` column
    m["fold"] = m["patient_id"].map(fold_of).apply(lambda x: "" if pd.isna(x) else str(int(x)))
    m.to_csv(C.MANIFEST_PATH, index=False)
    print(f"\n=== {kfolds}-fold CV assignment (patient-level, stratified) ===")
    eyes = m[m["usable"] == True].drop_duplicates(["patient_id", "eye"])  # noqa: E712
    print(pd.crosstab(eyes["fold"], eyes["dr_label"], margins=True).to_string())
    assert (m.loc[m["usable"] == True, "fold"] == "").sum() == 0, "usable rows missing a fold"  # noqa: E712
    print(f"Folds written to manifest 'fold' column (seed {SEED}).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kfolds", type=int, default=0,
                    help="If >0, ALSO write a patient-level stratified K-fold 'fold' column "
                         "for cross-validation (does not change train/val/test).")
    args = ap.parse_args()

    m = pd.read_csv(C.MANIFEST_PATH)
    u = m[m["usable"] == True].copy()  # noqa: E712
    u["dr_label"] = u["dr_label"].astype(int)

    # patient-level table with worst-eye grade
    pat = u.groupby("patient_id")["dr_label"].max().reset_index()
    pat.columns = ["patient_id", "worst_grade"]
    print(f"Patients with usable data: {len(pat)}")
    print("Worst-eye grade distribution (patients):")
    print(pat["worst_grade"].value_counts().sort_index().to_string())

    # 1) hold out test
    train_val, test = train_test_split(
        pat, test_size=TEST_FRAC, random_state=SEED, stratify=pat["worst_grade"])
    # 2) carve val out of the remainder so val is VAL_FRAC of the WHOLE
    val_rel = VAL_FRAC / (1.0 - TEST_FRAC)
    train, val = train_test_split(
        train_val, test_size=val_rel, random_state=SEED, stratify=train_val["worst_grade"])

    split_of = {}
    for pid in train["patient_id"]:
        split_of[pid] = "train"
    for pid in val["patient_id"]:
        split_of[pid] = "val"
    for pid in test["patient_id"]:
        split_of[pid] = "test"

    # assert disjoint
    s_tr, s_va, s_te = set(train.patient_id), set(val.patient_id), set(test.patient_id)
    assert s_tr.isdisjoint(s_va) and s_tr.isdisjoint(s_te) and s_va.isdisjoint(s_te), "OVERLAP!"
    assert len(s_tr) + len(s_va) + len(s_te) == len(pat)
    print(f"\nPatient split: train={len(s_tr)}  val={len(s_va)}  test={len(s_te)}  (disjoint OK)")

    # write split into manifest (only usable rows)
    m["split"] = m["patient_id"].map(split_of).fillna("")
    # rows that are usable must have a split; unusable rows stay ""
    bad = m[(m["usable"] == True) & (m["split"] == "")]  # noqa: E712
    assert len(bad) == 0, f"{len(bad)} usable rows without a split"
    m.to_csv(C.MANIFEST_PATH, index=False)

    # ---- report distributions ----
    print("\n=== IMAGE-level class distribution per split (usable) ===")
    uu = m[m["usable"] == True]  # noqa: E712
    print(pd.crosstab(uu["split"], uu["dr_label"], margins=True).to_string())

    print("\n=== EYE-level class distribution per split (unique eyes) ===")
    eyes = uu.drop_duplicates(["patient_id", "eye"])
    ct = pd.crosstab(eyes["split"], eyes["dr_label"], margins=True)
    print(ct.to_string())

    print("\n=== Binary referable (eye-level) per split ===")
    print(pd.crosstab(eyes["split"], eyes["referable"], margins=True).to_string())

    # near-empty minority warning
    print()
    for sp in ["train", "val", "test"]:
        e = eyes[eyes["split"] == sp]
        mins = {g: int((e["dr_label"] == g).sum()) for g in [2, 3]}
        if min(mins.values()) < 10:
            print(f"[WARN] split '{sp}' has few minority eyes: {mins} -> metrics unstable; "
                  f"consider k-fold CV.")
    print("\nSplit persisted to manifest with seed", SEED)

    if args.kfolds and args.kfolds > 1:
        m2 = pd.read_csv(C.MANIFEST_PATH)
        u2 = m2[m2["usable"] == True].copy()  # noqa: E712
        u2["dr_label"] = u2["dr_label"].astype(int)
        add_kfolds(m2, u2, args.kfolds)


if __name__ == "__main__":
    main()
