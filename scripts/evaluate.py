"""Step 3: clinical evaluation of the trained models from their saved predictions.

- operating threshold chosen on VALIDATION only (target sensitivity, and Youden for reference),
  then applied once to TEST;
- 95% confidence intervals by patient-level bootstrap;
- paired bootstrap of the AUROC difference between models;
- calibration (Brier, ECE), sensitivity by subtype (bacterial / viral), list of errors.

Usage: python scripts/evaluate.py --results results --target-sensitivity 0.98
Outputs: results/evaluation.md, results/evaluation.json, results/errors_<model>.csv,
         results/roc_calibration.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pneumonia.evaluation import (  # noqa: E402
    ci,
    confusion_metrics,
    global_metrics,
    patient_bootstrap,
    threshold_for_sensitivity,
    threshold_youden,
)


def fmt(x: float, lo: float | None = None, hi: float | None = None, pct: bool = True) -> str:
    f = (lambda v: f"{100 * v:.1f} %") if pct else (lambda v: f"{v:.4f}")
    return f"{f(x)} [{f(lo)} ; {f(hi)}]" if lo is not None else f(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--target-sensitivity", type=float, default=0.98)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    models = sorted(d.name for d in args.results.iterdir() if (d / "pred_test.csv").exists())
    val = {m: pd.read_csv(args.results / m / "pred_val.csv") for m in models}
    test = {m: pd.read_csv(args.results / m / "pred_test.csv") for m in models}
    report: dict = {"target_sensitivity": args.target_sensitivity, "models": {}}

    boots_auc = patient_bootstrap(test, roc_auc_score, n=args.n_boot)
    lines = [
        "# Step 3 — Clinical evaluation",
        "",
        f"Test set: {len(next(iter(test.values())))} images, "
        f"{next(iter(test.values()))['patient_id'].nunique()} patients. "
        f"Thresholds chosen on validation only. 95 % CI: patient bootstrap, n = {args.n_boot}.",
        "",
        "## Threshold-free metrics (test)",
        "",
        "| model | AUROC [95 % CI] | AUPRC | Brier | ECE |",
        "|---|---|---|---|---|",
    ]
    for m in models:
        y, p = test[m]["target"].to_numpy(), test[m]["prob"].to_numpy()
        g = global_metrics(y, p)
        lo, hi = ci(boots_auc[m])
        report["models"][m] = {"test_global": g, "test_auroc_ci": [lo, hi]}
        lines.append(
            f"| {m} | {g['auroc']:.4f} [{lo:.4f} ; {hi:.4f}] | {g['auprc']:.4f} | "
            f"{g['brier']:.4f} | {g['ece']:.4f} |"
        )

    lines += ["", "## Paired AUROC differences (test, same bootstrap patients)", ""]
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            d = boots_auc[a] - boots_auc[b]
            lo, hi = ci(d)
            verdict = "significant" if lo > 0 or hi < 0 else "not significant (CI contains 0)"
            lines.append(f"- {a} - {b}: {np.nanmean(d):+.4f} [{lo:+.4f} ; {hi:+.4f}] → {verdict}")
            report.setdefault("auroc_differences", {})[f"{a} - {b}"] = [float(lo), float(hi)]

    lines += [
        "",
        f"## Operating point: threshold set on validation for sensitivity ≥ "
        f"{args.target_sensitivity:.0%}, applied to test",
        "",
        "| model | threshold | sensitivity [95 % CI] | specificity [95 % CI] | PPV | NPV | "
        "missed pneumonias (FN) | false alarms (FP) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    youden_lines = []
    subtype_lines = []
    for m in models:
        yv, pv = val[m]["target"].to_numpy(), val[m]["prob"].to_numpy()
        yt, pt = test[m]["target"].to_numpy(), test[m]["prob"].to_numpy()
        t = threshold_for_sensitivity(yv, pv, args.target_sensitivity)
        cm = confusion_metrics(yt, pt, t)
        bs = patient_bootstrap(
            {m: test[m]}, lambda y, p, t=t: confusion_metrics(y, p, t)["sensitivity"], args.n_boot
        )[m]
        bp = patient_bootstrap(
            {m: test[m]}, lambda y, p, t=t: confusion_metrics(y, p, t)["specificity"], args.n_boot
        )[m]
        lines.append(
            f"| {m} | {t:.4f} | {fmt(cm['sensitivity'], *ci(bs))} | "
            f"{fmt(cm['specificity'], *ci(bp))} | {fmt(cm['ppv'])} | {fmt(cm['npv'])} | "
            f"{cm['fn']} | {cm['fp']} |"
        )
        ty = threshold_youden(yv, pv)
        cy = confusion_metrics(yt, pt, ty)
        c05 = confusion_metrics(yt, pt, 0.5)
        youden_lines.append(
            f"| {m} | {ty:.4f} | {fmt(cy['sensitivity'])} | {fmt(cy['specificity'])} | "
            f"{fmt(c05['sensitivity'])} | {fmt(c05['specificity'])} |"
        )
        df = test[m].assign(pred=(pt >= t).astype(int))
        sub = df[df.target == 1].groupby("subtype")["pred"].agg(["mean", "size"])
        subtype_lines.append(
            f"| {m} | "
            + " | ".join(
                f"{100 * sub.loc[s, 'mean']:.1f} % (n = {sub.loc[s, 'size']})"
                for s in ("bacteria", "virus")
            )
            + " |"
        )
        errors = df[df.pred != df.target].assign(
            error=lambda d: np.where(d.target == 1, "missed pneumonia (FN)", "false alarm (FP)")
        )
        errors.sort_values("prob").to_csv(args.results / f"errors_{m}.csv", index=False)
        report["models"][m].update(
            {
                "threshold_sensitivity_target": t,
                "test_at_threshold": cm,
                "test_sensitivity_ci": list(ci(bs)),
                "test_specificity_ci": list(ci(bp)),
                "threshold_youden": ty,
                "test_at_youden": cy,
                "test_at_0_5": c05,
                "sensitivity_by_subtype": sub["mean"].to_dict(),
            }
        )

    lines += [
        "",
        "## For reference: Youden threshold (val) and default 0.5",
        "",
        "| model | Youden threshold | sensitivity | specificity | sens. @0.5 | spec. @0.5 |",
        "|---|---|---|---|---|---|",
        *youden_lines,
        "",
        "## Sensitivity by pneumonia subtype (test, operating threshold)",
        "",
        "| model | bacterial | viral |",
        "|---|---|---|",
        *subtype_lines,
        "",
        "Errors per model: `errors_<model>.csv` (to inspect visually).",
        "",
    ]

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        for m in models:
            y, p = test[m]["target"].to_numpy(), test[m]["prob"].to_numpy()
            fpr, tpr, _ = roc_curve(y, p)
            ax1.plot(fpr, tpr, label=f"{m} (AUROC {roc_auc_score(y, p):.4f})")
            edges = np.linspace(0, 1, 11)
            idx = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
            xs = [p[idx == b].mean() for b in range(10) if (idx == b).any()]
            ys = [y[idx == b].mean() for b in range(10) if (idx == b).any()]
            ax2.plot(xs, ys, marker="o", label=m)
        ax1.plot([0, 1], [0, 1], "k:", lw=1)
        ax1.set(xlabel="1 - specificity", ylabel="sensitivity", title="ROC (test)")
        ax2.plot([0, 1], [0, 1], "k:", lw=1)
        ax2.set(
            xlabel="predicted probability", ylabel="observed frequency", title="Calibration (test)"
        )
        for ax in (ax1, ax2):
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.results / "roc_calibration.png", dpi=130)
        lines.append("![ROC and calibration](roc_calibration.png)")
    except ImportError:
        pass

    (args.results / "evaluation.md").write_text("\n".join(lines) + "\n")
    (args.results / "evaluation.json").write_text(json.dumps(report, indent=2, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
