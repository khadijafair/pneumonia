"""Step 1: index the dataset, audit the original split, build a patient-level split.

Usage:
  python scripts/prepare_data.py --download          # downloads with kagglehub
  python scripts/prepare_data.py --data-dir /path    # folder that contains chest_xray/
Outputs (data/splits/): index.csv, split.csv, duplicates.csv, report.json, report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pneumonia.data.index import build_index, find_dataset_root  # noqa: E402
from pneumonia.data.split import (  # noqa: E402
    audit_original_split,
    dedupe_exact,
    patient_split,
    split_summary,
)

KAGGLE_DATASET = "paultimothymooney/chest-xray-pneumonia"


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--download", action="store_true", help="download with kagglehub")
    src.add_argument("--data-dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/splits"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    if args.download:
        import kagglehub

        base = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    else:
        base = args.data_dir
    root = find_dataset_root(base)
    print(f"Dataset root: {root}")

    df = build_index(root)
    df.insert(0, "root", str(root))
    audit = audit_original_split(df)
    unparsed = int((~df["parsed"]).sum())

    kept, dropped = dedupe_exact(df)
    split_df = patient_split(kept, args.test_frac, args.val_frac, args.seed)
    summary = split_summary(split_df)
    orig_summary = split_summary(df, col="orig_split")

    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "index.csv", index=False)
    split_df.to_csv(args.out / "split.csv", index=False)
    dropped.to_csv(args.out / "duplicates.csv", index=False)

    report = {
        "dataset": KAGGLE_DATASET,
        "images_total": len(df),
        "patients_total": int(df["patient_id"].nunique()),
        "unparsed_filenames": unparsed,
        "exact_duplicates_removed": len(dropped),
        "original_split_audit": audit,
        "original_split": orig_summary.to_dict(orient="index"),
        "new_split": summary.to_dict(orient="index"),
        "seed": args.seed,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2, default=int))

    md = [
        "# Data report",
        "",
        f"- Images: {len(df)} | patients: {report['patients_total']} | "
        f"unparsed file names: {unparsed} | exact duplicates removed: {len(dropped)}",
        "",
        "## Original Kaggle split",
        "",
        orig_summary.to_markdown(),
        "",
        "Leakage audit of the original split:",
        "",
        *[f"- {k}: {v}" for k, v in audit.items() if k != "images_per_split"],
        "",
        f"## New patient-level split (seed {args.seed})",
        "",
        summary.to_markdown(),
        "",
    ]
    (args.out / "report.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
