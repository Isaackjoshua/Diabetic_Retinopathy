"""
Evaluation helpers for the DR / RETFound pipeline (Phase 5).

Works on the 4-class ORDINAL head (R0..R3) and derives the binary REFERABLE-DR
view (R2+) by collapsing probabilities: P(referable) = P(R2) + P(R3).

Reports EYE-level metrics (primary) by averaging the softmax over an eye's images,
and PATIENT-level metrics by worst-eye aggregation. Image-level is also available.

Selection metric = quadratic-weighted Cohen's kappa (QWK).
"""
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    cohen_kappa_score, roc_auc_score, average_precision_score, f1_score,
    accuracy_score, confusion_matrix, recall_score, roc_curve, precision_recall_curve,
)

N_CLASSES = 4
REFERABLE_FROM = 2  # grades >= 2 are referable


# ---------------------------------------------------------------- prediction
# Test-time augmentation ops on already-normalised tensor batches (B,C,H,W).
# Flips are label-preserving for fundus images (no orientation-encoded text).
_TTA_OPS = {
    "identity": lambda x: x,
    "hflip": lambda x: torch.flip(x, dims=[3]),
    "vflip": lambda x: torch.flip(x, dims=[2]),
    "hvflip": lambda x: torch.flip(x, dims=[2, 3]),
}


@torch.no_grad()
def predict(model, loader, device, amp=True, tta=None):
    """Return (y_true, y_prob) in loader (== dataset.samples) order.

    Loader MUST use a non-shuffling sampler so predictions align with
    loader.dataset.samples for eye/patient aggregation.

    tta: None/[] or list of view names from _TTA_OPS. When given, the softmax
    is averaged over the augmented views (test-time augmentation). Default
    single-view "identity" -> identical to no TTA.
    """
    views = list(tta) if tta else ["identity"]
    for v in views:
        if v not in _TTA_OPS:
            raise ValueError(f"unknown TTA view '{v}'; choose from {list(_TTA_OPS)}")
    model.eval()
    ys, ps = [], []
    for batch in loader:
        images, target = batch[0].to(device, non_blocking=True), batch[-1]
        prob = None
        for v in views:
            with torch.cuda.amp.autocast(enabled=amp):
                out = model(_TTA_OPS[v](images))
            p = F.softmax(out.float(), dim=1)
            prob = p if prob is None else prob + p
        prob = prob / len(views)
        ps.append(prob.cpu().numpy())
        ys.append(target.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def parse_pid_eye(path):
    """Symlink basename is '<pid>_<eye>_<orig>' (see materialize_imagefolder.py)."""
    b = os.path.basename(path)
    parts = b.split("_")
    return parts[0], parts[1]


# ---------------------------------------------------------------- aggregation
def aggregate(paths, y_true, y_prob, level="eye"):
    """Aggregate image-level probs to eye or patient level.

    eye     : mean softmax over the eye's images; label = the (single) eye grade.
    patient : worst-eye rule -- take the eye with the highest expected grade;
              report that eye's prob vector and the patient's worst true grade.
    Returns (y_true_agg, y_prob_agg).
    """
    # group to eye first
    eye_key, eye_true, eye_prob = {}, {}, {}
    acc = {}
    for p, yt, yp in zip(paths, y_true, y_prob):
        pid, eye = parse_pid_eye(p)
        k = (pid, eye)
        acc.setdefault(k, []).append(yp)
        eye_true[k] = yt
    for k, plist in acc.items():
        eye_prob[k] = np.mean(plist, axis=0)
    if level == "eye":
        keys = sorted(eye_prob)
        return (np.array([eye_true[k] for k in keys]),
                np.stack([eye_prob[k] for k in keys]))
    # patient worst-eye
    grades = np.arange(N_CLASSES)
    by_pat = {}
    for (pid, eye), prob in eye_prob.items():
        exp_grade = float((prob * grades).sum())          # expected grade for ranking eyes
        by_pat.setdefault(pid, []).append((exp_grade, eye_true[(pid, eye)], prob))
    yts, yps = [], []
    for pid, lst in by_pat.items():
        # worst eye by predicted expected grade
        _, _, worst_prob = max(lst, key=lambda t: t[0])
        worst_true = max(t[1] for t in lst)               # patient's worst true grade
        yts.append(worst_true); yps.append(worst_prob)
    return np.array(yts), np.stack(yps)


# ---------------------------------------------------------------- metrics
def _sens_spec(cm):
    out = {}
    total = cm.sum()
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = total - tp - fn - fp
        sens = float(tp / (tp + fn)) if (tp + fn) else float("nan")   # recall
        prec = float(tp / (tp + fp)) if (tp + fp) else float("nan")
        f1 = (2 * prec * sens / (prec + sens)
              if (prec == prec and sens == sens and (prec + sens) > 0) else float("nan"))
        out[c] = {
            "sensitivity": sens,            # == recall
            "recall": sens,
            "precision": prec,
            "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
            "f1": f1,
            "support": int(tp + fn),
        }
    return out


def macro_sens_spec(y_true, y_pred):
    """(macro_sensitivity, macro_specificity) over classes present in y_true."""
    cm = confusion_matrix(np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int),
                          labels=list(range(N_CLASSES)))
    ss = _sens_spec(cm)
    sens = [v["sensitivity"] for v in ss.values() if v["support"] > 0]
    spec = [v["specificity"] for v in ss.values() if v["support"] > 0]
    return (float(np.nanmean(sens)) if sens else float("nan"),
            float(np.nanmean(spec)) if spec else float("nan"))


def compute_metrics(y_true, y_prob, class_names=None, binary_op_point=0.5):
    """Full metric bundle for one aggregation level."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = y_prob.argmax(1)
    present = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    labels = list(range(N_CLASSES))

    res = {}
    res["n"] = int(len(y_true))
    res["accuracy"] = float(accuracy_score(y_true, y_pred))
    res["qwk"] = float(cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=labels))
    res["kappa_unweighted"] = float(cohen_kappa_score(y_true, y_pred, labels=labels))
    res["macro_f1"] = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))

    # macro AUROC / AUPRC (OvR) -- only over classes present in y_true
    try:
        yt_oh = np.eye(N_CLASSES)[y_true]
        cols = [c for c in labels if yt_oh[:, c].sum() > 0]
        res["macro_auroc_ovr"] = float(roc_auc_score(yt_oh[:, cols], y_prob[:, cols],
                                                      average="macro", multi_class="ovr"))
        res["macro_auprc"] = float(average_precision_score(yt_oh[:, cols], y_prob[:, cols],
                                                           average="macro"))
    except Exception as e:  # noqa
        res["macro_auroc_ovr"] = float("nan"); res["macro_auprc"] = float("nan")
        res["auroc_error"] = str(e)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    res["confusion_matrix"] = cm.tolist()
    res["per_class"] = _sens_spec(cm)
    # macro sensitivity / specificity = unweighted mean of per-class values,
    # averaged only over classes that actually occur in y_true (support > 0).
    sens = [v["sensitivity"] for v in res["per_class"].values() if v["support"] > 0]
    spec = [v["specificity"] for v in res["per_class"].values() if v["support"] > 0]
    prec = [v["precision"] for v in res["per_class"].values() if v["support"] > 0]
    res["macro_sensitivity"] = float(np.nanmean(sens)) if sens else float("nan")   # == macro recall
    res["macro_recall"] = res["macro_sensitivity"]
    res["macro_specificity"] = float(np.nanmean(spec)) if spec else float("nan")
    res["macro_precision"] = float(np.nanmean(prec)) if prec else float("nan")

    # ---- derived BINARY referable DR (R2+) ----
    b_true = (y_true >= REFERABLE_FROM).astype(int)
    b_score = y_prob[:, REFERABLE_FROM:].sum(1)          # P(R2)+P(R3)
    b = {"n_pos": int(b_true.sum()), "n_neg": int((1 - b_true).sum())}
    if b_true.sum() > 0 and (1 - b_true).sum() > 0:
        b["auroc"] = float(roc_auc_score(b_true, b_score))
        b["auprc"] = float(average_precision_score(b_true, b_score))
        b_pred = (b_score >= binary_op_point).astype(int)
        bcm = confusion_matrix(b_true, b_pred, labels=[0, 1])
        tn, fp, fn, tp = bcm.ravel()
        b["operating_point"] = binary_op_point
        b["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else float("nan")
        b["specificity"] = float(tn / (tn + fp)) if (tn + fp) else float("nan")
        b["confusion_matrix"] = bcm.tolist()
    res["binary_referable"] = b
    return res


def save_curves(y_true, y_prob, out_dir, tag):
    """Save ROC + PR (binary referable) and normalized confusion matrix plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    y_true = np.asarray(y_true).astype(int)
    b_true = (y_true >= REFERABLE_FROM).astype(int)
    b_score = y_prob[:, REFERABLE_FROM:].sum(1)

    if b_true.sum() and (1 - b_true).sum():
        fpr, tpr, _ = roc_curve(b_true, b_score)
        prec, rec, _ = precision_recall_curve(b_true, b_score)
        auroc = roc_auc_score(b_true, b_score)
        auprc = average_precision_score(b_true, b_score)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        ax[0].plot(fpr, tpr, label=f"AUROC={auroc:.3f}"); ax[0].plot([0, 1], [0, 1], "k--", lw=.7)
        ax[0].set_xlabel("1 - specificity"); ax[0].set_ylabel("sensitivity")
        ax[0].set_title(f"ROC — referable DR ({tag})"); ax[0].legend(loc="lower right")
        ax[1].plot(rec, prec, label=f"AUPRC={auprc:.3f}")
        ax[1].axhline(b_true.mean(), ls="--", c="grey", lw=.7, label=f"prevalence={b_true.mean():.3f}")
        ax[1].set_xlabel("recall (sensitivity)"); ax[1].set_ylabel("precision (PPV)")
        ax[1].set_title(f"PR — referable DR ({tag})"); ax[1].legend(loc="upper right")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"roc_pr_{tag}.png"), dpi=150)
        plt.close(fig)

    # confusion matrix (ordinal), row-normalized
    cm = confusion_matrix(y_true, y_prob.argmax(1), labels=list(range(N_CLASSES))).astype(float)
    cmn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, f"{cmn[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(["R0", "R1", "R2", "R3"]); ax.set_yticklabels(["R0", "R1", "R2", "R3"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"Confusion (row-norm) — {tag}")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"confusion_{tag}.png"), dpi=150)
    plt.close(fig)


def class_names_default():
    return ["R0", "R1", "R2", "R3"]


def full_report(paths, y_true, y_prob, out_dir):
    """Compute image/eye/patient metrics, save JSON + curves. Returns dict."""
    os.makedirs(out_dir, exist_ok=True)
    report = {}
    report["image_level"] = compute_metrics(y_true, y_prob)
    yt_e, yp_e = aggregate(paths, y_true, y_prob, "eye")
    report["eye_level"] = compute_metrics(yt_e, yp_e)
    yt_p, yp_p = aggregate(paths, y_true, y_prob, "patient")
    report["patient_level"] = compute_metrics(yt_p, yp_p)
    save_curves(yt_e, yp_e, out_dir, "eye")
    save_curves(yt_p, yp_p, out_dir, "patient")
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report
