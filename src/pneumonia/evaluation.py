"""Clinical evaluation: threshold chosen on validation, patient-level bootstrap CIs, calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def confusion_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = p >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fn = int(np.sum(~pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    tn = int(np.sum(~pred & (y == 0)))

    def div(a, b):
        return a / b if b else float("nan")

    return {
        "threshold": float(threshold),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "sensitivity": div(tp, tp + fn),  # recall: pneumonias found
        "specificity": div(tn, tn + fp),  # healthy children correctly cleared
        "ppv": div(tp, tp + fp),
        "npv": div(tn, tn + fn),
        "accuracy": div(tp + tn, len(y)),
    }


def threshold_for_sensitivity(y: np.ndarray, p: np.ndarray, target: float) -> float:
    """Highest threshold with sensitivity >= target (best specificity under that constraint)."""
    pos = np.sort(p[y == 1])[::-1]
    if len(pos) == 0:
        raise ValueError("no positive case")
    k = int(np.ceil(target * len(pos)))  # need at least k positives at or above the threshold
    return float(pos[max(k, 1) - 1])


def threshold_youden(y: np.ndarray, p: np.ndarray) -> float:
    best, best_j = 0.5, -1.0
    for t in np.unique(p):
        m = confusion_metrics(y, p, t)
        j = m["sensitivity"] + m["specificity"] - 1
        if j > best_j:
            best, best_j = float(t), j
    return best


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def global_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": expected_calibration_error(y, p),
    }


def patient_bootstrap(
    frames: dict[str, pd.DataFrame],
    stat,
    n: int = 2000,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Resample PATIENTS (not images) with replacement, identically for every model.

    frames: model name -> predictions (path, patient_id, target, prob), same images for all.
    stat(y, p) -> float. Returns model name -> array of n bootstrap values.
    """
    ref = next(iter(frames.values()))
    patients = ref["patient_id"].unique()
    rows_by_patient = ref.groupby("patient_id").indices
    aligned = {k: v.set_index("path").loc[ref["path"]].reset_index() for k, v in frames.items()}
    rng = np.random.default_rng(seed)
    out = {k: np.empty(n) for k in frames}
    for i in range(n):
        sample = rng.choice(patients, size=len(patients), replace=True)
        rows = np.concatenate([rows_by_patient[s] for s in sample])
        for k, df in aligned.items():
            y = df["target"].to_numpy()[rows]
            p = df["prob"].to_numpy()[rows]
            out[k][i] = stat(y, p) if 0 < y.mean() < 1 else np.nan
    return out


def ci(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    v = values[~np.isnan(values)]
    return float(np.quantile(v, a)), float(np.quantile(v, 1 - a))
