"""
Phase 3 -- materialize RETFound's ImageFolder layout via symlinks.

  outputs/dr_imagefolder/{train,val,test}/<class_name>/<pid>_<eye>_<orig>.jpg

Class folder names are chosen so torchvision.datasets.ImageFolder's alphabetical
ordering == our intended integer labels 0..3:
  R0_no_dr(0) < R1_mild(1) < R2_moderate_severe(2) < R3_proliferative(3)
The mapping is also written to outputs/class_mapping.json and asserted.

Run:  python pipeline/materialize_imagefolder.py
"""
import os
import re
import sys
import json
import shutil
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

DATA_ROOT = os.path.join(C.OUT_DIR, "dr_imagefolder")
CLASS_MAP_PATH = os.path.join(C.OUT_DIR, "class_mapping.json")


def sanitize(name):
    name = name.replace(" ", "_")
    name = re.sub(r"[^0-9A-Za-z._-]", "", name)
    return name


def main():
    m = pd.read_csv(C.MANIFEST_PATH)
    u = m[(m["usable"] == True) & (m["split"] != "")].copy()  # noqa: E712
    u["dr_label"] = u["dr_label"].astype(int)

    # verify alphabetical class order == intended index
    names = [C.ORDINAL_CLASS_NAMES[i] for i in range(len(C.ORDINAL_CLASS_NAMES))]
    assert names == sorted(names), f"class names not in sorted order: {names}"
    class_map = {C.ORDINAL_CLASS_NAMES[i]: i for i in range(len(C.ORDINAL_CLASS_NAMES))}

    if os.path.isdir(DATA_ROOT):
        shutil.rmtree(DATA_ROOT)

    made, collisions = 0, 0
    seen = set()
    for _, r in u.iterrows():
        split = r["split"]
        cls = C.ORDINAL_CLASS_NAMES[int(r["dr_label"])]
        d = os.path.join(DATA_ROOT, split, cls)
        os.makedirs(d, exist_ok=True)
        orig = os.path.basename(r["image_path"])
        link_name = sanitize(f"{r['patient_id']}_{r['eye']}_{orig}")
        link = os.path.join(d, link_name)
        if link in seen:
            collisions += 1
            base, ext = os.path.splitext(link)
            link = f"{base}_{collisions}{ext}"
        seen.add(link)
        os.symlink(os.path.abspath(r["image_path"]), link)
        made += 1

    with open(CLASS_MAP_PATH, "w") as f:
        json.dump({"ordinal_class_to_index": class_map,
                   "index_to_class": C.ORDINAL_CLASS_NAMES,
                   "note": "R3A and R3S collapsed into R3_proliferative"}, f, indent=2)

    print(f"Symlinks created: {made}  (name collisions auto-suffixed: {collisions})")
    print(f"Class mapping: {class_map}")

    # ---- re-verify ----
    print("\n=== per-split/per-class symlink counts (built) vs manifest ===")
    ok = True
    for split in ["train", "val", "test"]:
        for cls, idx in class_map.items():
            p = os.path.join(DATA_ROOT, split, cls)
            built = len(os.listdir(p)) if os.path.isdir(p) else 0
            exp = int(((u["split"] == split) & (u["dr_label"] == idx)).sum())
            flag = "" if built == exp else "  <-- MISMATCH"
            if built != exp:
                ok = False
            print(f"  {split:5} {cls:22} built={built:5} manifest={exp:5}{flag}")
    assert ok, "counts mismatch"

    # ---- patient-overlap check from the symlink names themselves ----
    def patients_in(split):
        s = set()
        for cls in class_map:
            p = os.path.join(DATA_ROOT, split, cls)
            for fn in os.listdir(p):
                s.add(fn.split("_")[0])
        return s
    tr, va, te = patients_in("train"), patients_in("val"), patients_in("test")
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te), "PATIENT OVERLAP in ImageFolder!"
    print(f"\nPatient overlap check (from symlink names): train={len(tr)} val={len(va)} test={len(te)}  DISJOINT OK")
    print(f"ImageFolder root: {DATA_ROOT}")


if __name__ == "__main__":
    main()
