"""Generate a standalone DR evaluation notebook.

Parameterized so the baseline and each experiment share one tested code path.
  Baseline:      python pipeline/build_eval_notebook.py
  Experiment 01: python pipeline/build_eval_notebook.py --ckpt-subdir experiment01 \
                   --results-subdir results_exp01 --out experiment01_evaluation.ipynb \
                   --title "Experiment 01 (384px + focal + macro-sensitivity selection)"
"""
import os
import argparse
import nbformat as nbf


def build(title, ckpt_subdir, results_subdir, out_name, prereq_nb):
    nb = nbf.v4.new_notebook()
    cells = []
    def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
    def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

    md(f"""
# {title}

Standalone, comprehensive evaluation on the held-out **test** split. Loads the
**fine-tuned checkpoint** (not the gated pretrained weights), so this runs without
any Hugging Face access once training is done.

**Metrics** (at image, **eye** [primary], and patient [worst-eye] levels):
precision, recall (= sensitivity), specificity, F1, **AUROC** (per-class OvR + macro),
AUPRC, quadratic-weighted kappa (QWK), accuracy, full **confusion matrices**, plus the
binary **referable-DR (R2+)** view with an operating-point sweep.

**Prerequisite:** run `{prereq_nb}` first so `outputs/{ckpt_subdir}/checkpoint-best.pth` exists.
""")

    code(f"""
# ---- locate project root (this notebook lives in evaluation/) ----
import os, sys, json
HERE = os.getcwd()
PROJECT = HERE if os.path.isdir(os.path.join(HERE, "pipeline")) else os.path.dirname(HERE)
os.chdir(PROJECT)
sys.path.insert(0, os.path.join(PROJECT, "pipeline"))
sys.path.insert(0, os.path.join(PROJECT, "RETFound_repo"))
print("project root:", PROJECT)

import numpy as np, torch, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, average_precision_score,
                             roc_auc_score, classification_report, confusion_matrix)
import dr_train as T, dr_eval as E

CKPT = os.path.join(PROJECT, "outputs", "{ckpt_subdir}", "checkpoint-best.pth")
assert os.path.exists(CKPT), f"missing {{CKPT}} -- run {prereq_nb} first"
RESULTS = os.path.join(PROJECT, "evaluation", "{results_subdir}"); os.makedirs(RESULTS, exist_ok=True)
CLASS_NAMES = ["R0", "R1", "R2", "R3"]; NC = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
""")

    code(r"""
# ---- rebuild model architecture and load the FINE-TUNED weights ----
ckpt = torch.load(CKPT, map_location="cpu")
cfg = ckpt.get("config", {})
print("checkpoint from epoch", ckpt.get("epoch"),
      "| val_QWK", round(ckpt.get("val_qwk", float('nan')), 4),
      "| val_macro_sens", round(ckpt.get("val_macro_sensitivity", float('nan')), 4))
DATA_PATH = cfg.get("data_path", "outputs/dr_imagefolder_cache")
INPUT = cfg.get("input_size", 224)
EVAL_BS = 32 if INPUT <= 224 else 12   # smaller batch for high-res eval

args = T.make_args({**{
    "data_path": DATA_PATH, "nb_classes": NC, "input_size": INPUT,
    "finetune_id": "", "drop_path": cfg.get("drop_path", 0.2),
    "batch_size": EVAL_BS, "accum_iter": 1, "epochs": 1, "warmup_epochs": 0,
    "blr": 5e-3, "layer_decay": 0.65, "weight_decay": 0.05, "min_lr": 1e-6,
    "output_dir": RESULTS, "seed": 42, "num_workers": 10,
}})
model = T.build_model_arch(args)
missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
assert not missing, f"unexpected missing keys: {missing}"
model.to(device).eval()
print(f"loaded fine-tuned weights OK | input_size={INPUT} eval_batch={EVAL_BS}")
""")

    code(r"""
# ---- predict on the TEST split (order aligned to dataset.samples) ----
(_, _, ds_te), (_, _, dl_te) = T.build_loaders(args, shuffle_train=False)
assert ds_te.class_to_idx == json.load(open("outputs/class_mapping.json"))["ordinal_class_to_index"]
test_paths = [p for p, _ in ds_te.samples]
y_true, y_prob = E.predict(model, dl_te, device)
y_pred = y_prob.argmax(1)
print(f"test images: {len(y_true)}  |  eyes: {len(set(E.parse_pid_eye(p) for p in test_paths))}")
""")

    code(r"""
# ---- full metric bundle at image / eye / patient levels (saved to results/metrics.json) ----
report = E.full_report(test_paths, y_true, y_prob, RESULTS)

rows = []
for lvl in ["image_level", "eye_level", "patient_level"]:
    r = report[lvl]; b = r["binary_referable"]
    rows.append({
        "level": lvl.replace("_level", ""), "n": r["n"],
        "QWK": r["qwk"], "accuracy": r["accuracy"],
        "macro_AUROC": r["macro_auroc_ovr"], "macro_AUPRC": r["macro_auprc"],
        "macro_precision": r["macro_precision"], "macro_recall(sens)": r["macro_sensitivity"],
        "macro_specificity": r["macro_specificity"], "macro_F1": r["macro_f1"],
        "referable_AUROC": b.get("auroc"), "referable_AUPRC": b.get("auprc"),
        "referable_sens": b.get("sensitivity"), "referable_spec": b.get("specificity"),
    })
summary = pd.DataFrame(rows).set_index("level")
summary.to_csv(os.path.join(RESULTS, "summary_metrics.csv"))
summary.round(4)
""")

    code(r"""
# ---- per-class table (eye level): precision / recall / specificity / F1 / support ----
eye = report["eye_level"]
pc = pd.DataFrame(eye["per_class"]).T
pc.index = CLASS_NAMES
pc = pc[["precision", "recall", "specificity", "f1", "support"]]
pc.loc["macro"] = [eye["macro_precision"], eye["macro_sensitivity"],
                   eye["macro_specificity"], eye["macro_f1"], pc["support"].sum()]
print("EYE-LEVEL per-class metrics:")
pc.round(4)
""")

    code(r"""
# ---- sklearn classification_report (eye level), for a familiar view ----
yt_e, yp_e = E.aggregate(test_paths, y_true, y_prob, "eye")
print("EYE-LEVEL classification report\n")
print(classification_report(yt_e, yp_e.argmax(1), labels=list(range(NC)),
                            target_names=CLASS_NAMES, zero_division=0, digits=4))
""")

    code(r"""
# ---- confusion matrices: counts + row-normalized (eye and patient) ----
def plot_cm(ax, y, p, title, norm):
    cm = confusion_matrix(y, p, labels=list(range(NC))).astype(float)
    disp = cm / np.clip(cm.sum(1, keepdims=True), 1, None) if norm else cm
    im = ax.imshow(disp, cmap="Blues", vmin=0, vmax=disp.max() if disp.max() else 1)
    for i in range(NC):
        for j in range(NC):
            txt = f"{disp[i,j]:.2f}" if norm else f"{int(cm[i,j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="white" if disp[i, j] > 0.5 * disp.max() else "black")
    ax.set_xticks(range(NC)); ax.set_yticks(range(NC))
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)

yt_p, yp_p = E.aggregate(test_paths, y_true, y_prob, "patient")
fig, ax = plt.subplots(2, 2, figsize=(10, 9))
plot_cm(ax[0, 0], yt_e, yp_e.argmax(1), "Eye — counts", False)
plot_cm(ax[0, 1], yt_e, yp_e.argmax(1), "Eye — row-normalized", True)
plot_cm(ax[1, 0], yt_p, yp_p.argmax(1), "Patient — counts", False)
plot_cm(ax[1, 1], yt_p, yp_p.argmax(1), "Patient — row-normalized", True)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "confusion_matrices.png"), dpi=150); plt.show()
""")

    code(r"""
# ---- ROC curves: per-class one-vs-rest + macro + binary referable (eye level) ----
yt_oh = np.eye(NC)[yt_e]
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for c in range(NC):
    if yt_oh[:, c].sum() == 0:
        continue
    fpr, tpr, _ = roc_curve(yt_oh[:, c], yp_e[:, c])
    ax[0].plot(fpr, tpr, label=f"{CLASS_NAMES[c]} (AUC={auc(fpr,tpr):.3f})")
cols = [c for c in range(NC) if yt_oh[:, c].sum() > 0]
macro_auc = roc_auc_score(yt_oh[:, cols], yp_e[:, cols], average="macro", multi_class="ovr")
ax[0].plot([0, 1], [0, 1], "k--", lw=.7)
ax[0].set_title(f"ROC per class (OvR) — eye | macro AUROC={macro_auc:.3f}")
ax[0].set_xlabel("1 - specificity"); ax[0].set_ylabel("sensitivity"); ax[0].legend(fontsize=8)

b_true = (yt_e >= 2).astype(int); b_score = yp_e[:, 2:].sum(1)
fpr, tpr, _ = roc_curve(b_true, b_score)
ax[1].plot(fpr, tpr, label=f"referable DR (AUROC={roc_auc_score(b_true,b_score):.3f})", color="C3")
ax[1].plot([0, 1], [0, 1], "k--", lw=.7)
ax[1].set_title("ROC — binary referable DR (R2+) — eye")
ax[1].set_xlabel("1 - specificity"); ax[1].set_ylabel("sensitivity"); ax[1].legend()
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "roc_curves.png"), dpi=150); plt.show()
""")

    code(r"""
# ---- Precision-Recall curves: per-class + binary referable (eye level) ----
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for c in range(NC):
    if yt_oh[:, c].sum() == 0:
        continue
    pr, rc, _ = precision_recall_curve(yt_oh[:, c], yp_e[:, c])
    ap = average_precision_score(yt_oh[:, c], yp_e[:, c])
    ax[0].plot(rc, pr, label=f"{CLASS_NAMES[c]} (AP={ap:.3f})")
ax[0].set_title("PR per class (OvR) — eye"); ax[0].set_xlabel("recall"); ax[0].set_ylabel("precision"); ax[0].legend(fontsize=8)

pr, rc, _ = precision_recall_curve(b_true, b_score)
ax[1].plot(rc, pr, color="C3", label=f"referable DR (AUPRC={average_precision_score(b_true,b_score):.3f})")
ax[1].axhline(b_true.mean(), ls="--", c="grey", lw=.7, label=f"prevalence={b_true.mean():.3f}")
ax[1].set_title("PR — binary referable DR (R2+) — eye"); ax[1].set_xlabel("recall (sensitivity)")
ax[1].set_ylabel("precision (PPV)"); ax[1].legend()
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "pr_curves.png"), dpi=150); plt.show()
""")

    code(r"""
# ---- per-class metric bar chart (eye level) ----
metrics = ["precision", "recall", "specificity", "f1"]
vals = {m: [eye["per_class"][c][m] for c in range(NC)] for m in metrics}
x = np.arange(NC); w = 0.2
fig, ax = plt.subplots(figsize=(9, 4.5))
for i, m in enumerate(metrics):
    ax.bar(x + (i - 1.5) * w, vals[m], w, label=m)
ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES); ax.set_ylim(0, 1.05)
ax.set_title("Per-class metrics (eye level)"); ax.set_ylabel("score"); ax.legend(ncol=4, fontsize=9)
for i, m in enumerate(metrics):
    for j in range(NC):
        v = vals[m][j]
        if v == v:
            ax.text(x[j] + (i - 1.5) * w, v + 0.01, f"{v:.2f}", ha="center", fontsize=7, rotation=90)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, "per_class_bars.png"), dpi=150); plt.show()
""")

    code(r"""
# ---- operating-point sweep for referable DR (choose a sensitivity target) ----
from sklearn.metrics import confusion_matrix as cmat
rows = []
for thr in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    pred = (b_score >= thr).astype(int)
    tn, fp, fn, tp = cmat(b_true, pred, labels=[0, 1]).ravel()
    rows.append({"threshold": thr,
                 "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
                 "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
                 "precision(PPV)": tp / (tp + fp) if (tp + fp) else float("nan"),
                 "TP": tp, "FP": fp, "FN": fn, "TN": tn})
op = pd.DataFrame(rows).set_index("threshold")
op.to_csv(os.path.join(RESULTS, "referable_operating_points.csv"))
print("Referable-DR operating points (eye level):")
op.round(4)
""")

    md(r"""
## Notes on interpretation
- **Eye level is primary**; patient level uses worst-eye aggregation. Image level is shown for completeness.
- Minority classes are small in the test split (~35 R2, ~22 R3 eyes) — per-class precision/recall
  for R2/R3 and the referable-DR PR curve rest on few positives; treat them as indicative and prefer
  k-fold CV for a firm estimate.
- For screening, pick the operating point from the sweep above by the **sensitivity** you require for
  referable DR, and report the corresponding specificity/PPV — do not default to 0.5.
- All figures and tables are saved under this notebook's results folder.
""")

    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"display_name": "retfound", "language": "python", "name": "python3"},
                      "language_info": {"name": "python"}}
    proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(proj, "evaluation"); os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, out_name)
    nbf.write(nb, out)
    print("wrote", out, "with", len(cells), "cells")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default="Diabetic Retinopathy — model evaluation")
    ap.add_argument("--ckpt-subdir", default="train_run")
    ap.add_argument("--results-subdir", default="results")
    ap.add_argument("--out", default="DR_evaluation.ipynb")
    ap.add_argument("--prereq-nb", default="RETFound_DR_finetune.ipynb")
    a = ap.parse_args()
    build(a.title, a.ckpt_subdir, a.results_subdir, a.out, a.prereq_nb)
