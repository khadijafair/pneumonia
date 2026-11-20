"""Step 5a: export a trained Lightning checkpoint to model.safetensors + config.json.

Only the network weights are kept (no optimizer state), in safetensors format (no pickle,
safe to load). The operating threshold is recomputed from the validation predictions of the
same run, with the same rule as the evaluation (sensitivity >= 98 %).

  python scripts/export_model.py --ckpt densenet121.ckpt \
      --pred-val results_rsna/densenet121/pred_val.csv \
      --external results_rsna/evaluation_external.json --out export/densenet121
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import timm
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pneumonia.evaluation import threshold_for_sensitivity  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--pred-val", type=Path, required=True)
    ap.add_argument("--external", type=Path, default=None, help="evaluation_external.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--target-sensitivity", type=float, default=0.98)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)  # our own file
    hp = ckpt["hyper_parameters"]
    state = {k[len("net.") :]: v for k, v in ckpt["state_dict"].items() if k.startswith("net.")}
    model = timm.create_model(hp["name"], pretrained=False, num_classes=1)
    model.load_state_dict(state)  # strict: fails if anything is missing

    val = pd.read_csv(args.pred_val)
    thr = threshold_for_sensitivity(
        val["target"].to_numpy(), val["prob"].to_numpy(), args.target_sensitivity
    )
    cfg = {
        "timm_name": hp["name"],
        "img_size": args.img_size,
        "mean": list(model.pretrained_cfg["mean"]),
        "std": list(model.pretrained_cfg["std"]),
        "threshold": round(float(thr), 4),
        "threshold_rule": f"validation sensitivity >= {args.target_sensitivity:.0%} (children)",
    }
    if args.external:
        ext = json.loads(args.external.read_text())["models"].get(Path(args.ckpt).stem, {})
        if ext:
            cfg["metrics"] = {
                "internal_test_auroc": round(ext["internal_test"]["auroc"], 4),
                "rsna_adults_auroc": round(ext["rsna_main"]["auroc"], 4),
                "rsna_sensitivity_at_threshold": round(ext["rsna_main"]["sensitivity"], 4),
                "rsna_specificity_at_threshold": round(ext["rsna_main"]["specificity"], 4),
            }
    args.out.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: v.contiguous() for k, v in model.state_dict().items()}, args.out / "model.safetensors"
    )
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2))
    size = (args.out / "model.safetensors").stat().st_size / 1e6
    print(json.dumps(cfg, indent=2), f"\nmodel.safetensors: {size:.1f} MB")


if __name__ == "__main__":
    main()
