import numpy as np
import pandas as pd
import pytest

from pneumonia.evaluation import (
    ci,
    confusion_metrics,
    expected_calibration_error,
    global_metrics,
    patient_bootstrap,
    threshold_for_sensitivity,
    threshold_youden,
)


def test_confusion_metrics_counts():
    y = np.array([1, 1, 1, 0, 0])
    p = np.array([0.9, 0.6, 0.2, 0.7, 0.1])
    m = confusion_metrics(y, p, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 1)
    assert m["sensitivity"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(1 / 2)


def test_threshold_for_sensitivity_reaches_target_and_is_highest():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(200), np.zeros(100)].astype(int)
    p = np.r_[rng.uniform(0.3, 1, 200), rng.uniform(0, 0.7, 100)]
    for target in (0.9, 0.95, 0.98, 1.0):
        t = threshold_for_sensitivity(y, p, target)
        assert confusion_metrics(y, p, t)["sensitivity"] >= target
        # any higher threshold breaks the target
        higher = p[p > t]
        if len(higher):
            assert confusion_metrics(y, p, higher.min())["sensitivity"] < target


def test_youden_on_separable_data():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    t = threshold_youden(y, p)
    m = confusion_metrics(y, p, t)
    assert m["sensitivity"] == 1 and m["specificity"] == 1


def test_calibration_perfect_vs_overconfident():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(int)  # calibrated by construction
    assert expected_calibration_error(y, p) < 0.02
    assert expected_calibration_error(y, (p > 0.5).astype(float)) > 0.15
    g = global_metrics(y, p)
    assert 0 < g["brier"] < 0.25 and 0.5 < g["auroc"] < 1


def test_patient_bootstrap_resamples_patients_and_is_paired():
    rows = []
    for pid in range(60):
        t = pid % 2
        for k in range(3):  # 3 images per patient
            rows.append({"path": f"{pid}_{k}", "patient_id": pid, "target": t})
    base = pd.DataFrame(rows)
    rng = np.random.default_rng(0)
    good = base.assign(prob=np.clip(base.target * 0.6 + rng.uniform(0, 0.4, len(base)), 0, 1))
    same = good.sample(frac=1, random_state=3)  # same predictions, shuffled rows
    from sklearn.metrics import roc_auc_score

    out = patient_bootstrap({"a": good, "b": same}, roc_auc_score, n=200)
    np.testing.assert_allclose(out["a"], out["b"])  # paired: identical resamples
    lo, hi = ci(out["a"])
    assert lo <= roc_auc_score(good.target, good.prob) <= hi
