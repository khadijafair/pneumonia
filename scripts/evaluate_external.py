"""Step 4b: external validation on RSNA (adults, other hospitals) vs internal test (children).

For each model:
  - the operating threshold is the one chosen on the INTERNAL validation set (as deployed);
  - main task: "Lung Opacity" vs "Normal"; images with another abnormality
    ("No Lung Opacity / Not Normal") are reported separately (false-alarm rate);
  - 95 % CI by bootstrap (one image per patient in RSNA);
  - local recalibration: Youden threshold chosen on 20 % of RSNA, evaluated on the other 80 %.

Usage: python scripts/evaluate_external.py --results results_rsna
Expects <results>/<model>/pred_val.csv, pred_test.csv (internal) and pred_rsna.csv.
Outputs: <results>/evaluation_external.md / .json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pneumonia.evaluation import (  # noqa: E402
    ci,
    confusion_metrics,
    global_metrics,
    patient_bootstrap,
    threshold_for_sensitivity,
    threshold_youden,
)


def pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results_rsna"))
    ap.add_argument("--target-sensitivity", type=float, default=0.98)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    models = sorted(d.name for d in args.results.iterdir() if (d / "pred_rsna.csv").exists())
    rsna = {m: pd.read_csv(args.results / m / "pred_rsna.csv") for m in models}
    ref = next(iter(rsna.values()))
    main_mask = ref["subtype"].isin(["rsna_opacity", "rsna_normal"]).to_numpy()
    counts = ref["subtype"].value_counts().to_dict()

    rng = np.random.default_rng(args.seed)
    calib = rng.uniform(size=len(ref)) < 0.2  # local recalibration subset

    report: dict = {"rsna_counts": counts, "models": {}}
    rows, rows_thr, rows_local, rows_classes = [], [], [], []
    main_frames = {m: rsna[m][main_mask].reset_index(drop=True) for m in models}
    boots = patient_bootstrap(main_frames, roc_auc_score, n=args.n_boot, seed=args.seed)

    for m in models:
        val = pd.read_csv(args.results / m / "pred_val.csv")
        test = pd.read_csv(args.results / m / "pred_test.csv")
        t = threshold_for_sensitivity(
            val["target"].to_numpy(), val["prob"].to_numpy(), args.target_sensitivity
        )
        yi, pi = test["target"].to_numpy(), test["prob"].to_numpy()
        internal = {**global_metrics(yi, pi), **confusion_metrics(yi, pi, t)}

        d = rsna[m]
        y, p = d["target"].to_numpy(), d["prob"].to_numpy()
        ym, pm = y[main_mask], p[main_mask]
        ext = {**global_metrics(ym, pm), **confusion_metrics(ym, pm, t)}
        ext_all_auroc = float(roc_auc_score(y, p))
        other = d["subtype"] == "rsna_not_normal"
        fa_other = float((d.loc[other, "prob"] >= t).mean()) if other.any() else float("nan")

        lo, hi = ci(boots[m])
        rows.append(
            f"| {m} | {internal['auroc']:.4f} | {ext['auroc']:.4f} [{lo:.4f} ; {hi:.4f}] | "
            f"{ext_all_auroc:.4f} | {ext['brier']:.4f} | {ext['ece']:.4f} |"
        )
        rows_thr.append(
            f"| {m} | {t:.4f} | {pct(internal['sensitivity'])} / {pct(internal['specificity'])} | "
            f"{pct(ext['sensitivity'])} / {pct(ext['specificity'])} | {pct(fa_other)} |"
        )

        # local recalibration on 20 % of RSNA (main task), evaluated on the other 80 %
        cm_mask, ev_mask = calib & main_mask, ~calib & main_mask
        tl = threshold_youden(y[cm_mask], p[cm_mask])
        loc = confusion_metrics(y[ev_mask], p[ev_mask], tl)
        same = confusion_metrics(y[ev_mask], p[ev_mask], t)
        rows_local.append(
            f"| {m} | {t:.4f} | {pct(same['sensitivity'])} / {pct(same['specificity'])} | "
            f"{tl:.4f} | {pct(loc['sensitivity'])} / {pct(loc['specificity'])} |"
        )

        def pair_auc(a: str, b: str, d=d) -> float:
            s_ = d[d["subtype"].isin([a, b])]
            return float(roc_auc_score(s_["subtype"] == a, s_["prob"]))

        pairs = {
            "opacity_vs_normal": pair_auc("rsna_opacity", "rsna_normal"),
            "other_vs_normal": pair_auc("rsna_not_normal", "rsna_normal"),
            "opacity_vs_other": pair_auc("rsna_opacity", "rsna_not_normal"),
        }
        rows_classes.append(
            "| " + m + " | " + " | ".join(f"{v:.4f}" for v in pairs.values()) + " |"
        )
        report["models"][m] = {
            "auroc_between_classes": pairs,
            "threshold_internal": t,
            "internal_test": internal,
            "rsna_main": ext,
            "rsna_main_auroc_ci": [lo, hi],
            "rsna_all_auroc": ext_all_auroc,
            "rsna_false_alarm_other_abnormality": fa_other,
            "local_threshold_youden": tl,
            "local_eval_80pct": loc,
            "internal_threshold_eval_80pct": same,
        }

    n_main = int(main_mask.sum())
    lines = [
        "# Step 4 — External validation on RSNA (adults) vs internal test (children)",
        "",
        f"RSNA images: {len(ref)} ({', '.join(f'{k}: {v}' for k, v in counts.items())}). "
        f"Main task = Lung Opacity vs Normal ({n_main} images). "
        f"95 % CI: bootstrap, n = {args.n_boot} (one image per patient).",
        "",
        "## Discrimination",
        "",
        "| model | AUROC internal (children) | AUROC RSNA opacity vs normal [95 % CI] | "
        "AUROC RSNA all images | Brier RSNA | ECE RSNA |",
        "|---|---|---|---|---|---|",
        *rows,
        "",
        f"## Operating threshold chosen on internal validation (sensitivity ≥ "
        f"{args.target_sensitivity:.0%})",
        "",
        "| model | threshold | internal sens. / spec. | RSNA sens. / spec. | "
        "false alarms on other abnormalities |",
        "|---|---|---|---|---|",
        *rows_thr,
        "",
        "## What does the model really detect? (AUROC between RSNA classes)",
        "",
        "| model | opacity vs normal | other abnormality vs normal | "
        "opacity vs other abnormality |",
        "|---|---|---|---|",
        *rows_classes,
        "",
        "## Local recalibration: Youden threshold set on 20 % of RSNA, evaluated on 80 %",
        "",
        "| model | internal threshold | sens. / spec. | local threshold | sens. / spec. |",
        "|---|---|---|---|---|",
        *rows_local,
        "",
    ]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve

        fig, axes = plt.subplots(1, len(models) + 1, figsize=(4.2 * (len(models) + 1), 3.8))
        for m, ax in zip(models, axes[1:], strict=True):
            d = rsna[m]
            for sub, name in (
                ("rsna_normal", "Normal"),
                ("rsna_not_normal", "Other abnormality"),
                ("rsna_opacity", "Lung opacity"),
            ):
                ax.hist(
                    d.loc[d.subtype == sub, "prob"],
                    bins=40,
                    range=(0, 1),
                    alpha=0.5,
                    density=True,
                    label=name,
                )
            ax.axvline(report["models"][m]["threshold_internal"], color="k", ls="--", lw=1)
            ax.set(title=f"{m}: RSNA probabilities", xlabel="predicted probability")
            ax.legend(fontsize=7)
        ax = axes[0]
        for m in models:
            test = pd.read_csv(args.results / m / "pred_test.csv")
            f1, t1, _ = roc_curve(test["target"], test["prob"])
            f2, t2, _ = roc_curve(rsna[m]["target"][main_mask], rsna[m]["prob"][main_mask])
            (line,) = ax.plot(f1, t1, lw=1, label=f"{m} internal")
            ax.plot(f2, t2, lw=1.5, ls="--", color=line.get_color(), label=f"{m} RSNA")
        ax.plot([0, 1], [0, 1], "k:", lw=1)
        ax.set(
            title="ROC: children (solid) vs adults (dashed)",
            xlabel="1 - specificity",
            ylabel="sensitivity",
        )
        ax.legend(fontsize=6, loc="lower right")
        fig.tight_layout()
        fig.savefig(args.results / "external.png", dpi=120)
        lines.append("![internal vs external](external.png)")
    except ImportError:
        pass

    (args.results / "evaluation_external.md").write_text("\n".join(lines))
    (args.results / "evaluation_external.json").write_text(
        json.dumps(report, indent=2, default=float)
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
