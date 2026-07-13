"""
Shared constants and helpers for the RETFound DR pipeline.

Key facts established in Phase 0 (see DATA_REPORT.md):
  * Image root: patient folder (4-digit numeric id) -> {LE, RE} -> *.JPG
  * Some patients have EXTRA "stray" subfolders (e.g. 1118/1118.24, 1139/New folder)
    holding more images of the SAME patient/visit. Eye is NOT encoded by these dirs,
    so laterality must come from the filename token in those cases.
  * Every image filename encodes the eye via a token _LE_ / _RE_ / _OD_ / _OS_
    (OD = oculus dexter = RIGHT = RE ; OS = oculus sinister = LEFT = LE).
    This is confirmed by patient 2406 whose RE/ folder holds _OD_ files and LE/ holds _OS_.
  * Labels live in Reading_Grades.xlsx -> Sheet2 (one row per patient, the canonical
    consensus grade; Sheet3 is a pivot of it, Sheet1 is the multi-stage grading history).
  * CSV `code` carries a trailing "_T" (e.g. 0019_T) that must be stripped to match the
    numeric folder name (0019). Label columns are explicitly named
    'Retinopathy (Left Eye)' and 'Retinopathy (Right Eye)' -> NO OD/OS ambiguity on the
    CSV side; Left Eye -> LE folder, Right Eye -> RE folder.
  * Grade scheme = UK NHS DESP: R0, R1, R2, R3A, R3S, U(=ungradable).
"""
import os
import re

# ---- paths ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_ROOT = os.path.join(PROJECT_ROOT, "Diabetic Retinopathy IMAGES 2")
XLSX_PATH = os.path.join(PROJECT_ROOT, "Reading_Grades.xlsx")
XLSX_SHEET = "Sheet2"
OUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MANIFEST_PATH = os.path.join(OUT_DIR, "manifest.csv")
DATA_REPORT_PATH = os.path.join(PROJECT_ROOT, "DATA_REPORT.md")

LABEL_LEFT_COL = "Retinopathy (Left Eye)"
LABEL_RIGHT_COL = "Retinopathy (Right Eye)"

# ---- label schemes ----
# 4-class ordinal (primary label). R3A and R3S both collapse to R3 (R3S has ~1 case).
GRADE_TO_ORDINAL = {"R0": 0, "R1": 1, "R2": 2, "R3A": 3, "R3S": 3}
ORDINAL_CLASS_NAMES = {0: "R0_no_dr", 1: "R1_mild", 2: "R2_moderate_severe", 3: "R3_proliferative"}
# Binary referable DR (secondary view): referable = R2 or worse.
GRADE_TO_BINARY = {"R0": 0, "R1": 0, "R2": 1, "R3A": 1, "R3S": 1}
BINARY_CLASS_NAMES = {0: "non_referable", 1: "referable_dr"}
UNGRADABLE = {"U"}

# ---- filename eye token ----
_EYE_TOKEN_RE = re.compile(r"_(LE|RE|OD|OS)_", re.IGNORECASE)
_TOKEN_TO_EYE = {"LE": "LE", "OS": "LE", "RE": "RE", "OD": "RE"}
# acquisition id = trailing alnum hash before optional " (n)" and extension
_ACQ_RE = re.compile(r"_([0-9A-Z]{8,})(?:\s*\(\d+\))?\.[A-Za-z]+$")


def strip_code_suffix(code):
    """CSV code -> folder id: strip a trailing _T (and normalise)."""
    s = str(code).strip()
    if s.endswith("_T"):
        s = s[:-2]
    return s


def eye_from_filename(name):
    """Return 'LE'/'RE' from the filename eye token, or None if absent."""
    m = _EYE_TOKEN_RE.search(name)
    if not m:
        return None
    return _TOKEN_TO_EYE[m.group(1).upper()]


def eye_from_path(rel_path):
    """Return 'LE'/'RE' if the path has an /LE/ or /RE/ component, else None."""
    parts = [p.upper() for p in rel_path.replace("\\", "/").split("/")]
    has_le = "LE" in parts
    has_re = "RE" in parts
    if has_le and not has_re:
        return "LE"
    if has_re and not has_le:
        return "RE"
    return None


def acquisition_id(name):
    """Trailing acquisition hash; files sharing it (e.g. ' (1)'/' (2)') are duplicates."""
    m = _ACQ_RE.search(name)
    return m.group(1) if m else None


def patient_id_from_path(rel_path):
    """Top-level folder = patient id."""
    return rel_path.replace("\\", "/").split("/")[0]
