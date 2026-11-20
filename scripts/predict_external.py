"""Step 4a: run trained checkpoints on the external RSNA set and save probabilities.

  python scripts/predict_external.py --rsna /kaggle/input --ckpt outputs/*/best.ckpt \
      --results results_rsna [--limit 8000]
Writes <results>/<model>/pred_rsna.csv (path, patient_id, label, target, subtype, boxes, prob).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pneumonia.data.dataset import ChestXrayDataset, build_transforms  # noqa: E402
from pneumonia.data.rsna import build_rsna_index, stratified_sample  # noqa: E402
from pneumonia.model import PneumoniaClassifier, backbone_norm  # noqa: E402


@torch.no_grad()
def predict(model: PneumoniaClassifier, loader: DataLoader, device: str) -> torch.Tensor:
    model.eval().to(device)
    out = []
    for x, _y, _i in loader:
        with torch.autocast(device_type="cuda", enabled=device == "cuda"):
            out.append(torch.sigmoid(model(x.to(device, non_blocking=True))).float().cpu())
    return torch.cat(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rsna", type=Path, required=True, help="folder containing the RSNA files")
    ap.add_argument("--ckpt", type=Path, nargs="+", required=True)
    ap.add_argument("--results", type=Path, default=Path("results_rsna"))
    ap.add_argument("--limit", type=int, default=None, help="stratified sample size")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    df, root = build_rsna_index(args.rsna)
    df = stratified_sample(df, args.limit)
    print(f"RSNA: {len(df)} images, by class: {df['subtype'].value_counts().to_dict()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for ckpt in args.ckpt:
        name = ckpt.parent.name
        model = PneumoniaClassifier.load_from_checkpoint(ckpt, map_location="cpu")
        mean, std = backbone_norm(model.net)
        loader = DataLoader(
            ChestXrayDataset(df, root, build_transforms(args.img_size, mean, std, train=False)),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device == "cuda",
        )
        prob = predict(model, loader, device).numpy()
        out = args.results / name
        out.mkdir(parents=True, exist_ok=True)
        df.assign(prob=prob).to_csv(out / "pred_rsna.csv", index=False)
        print(f"{name}: {out / 'pred_rsna.csv'}")


if __name__ == "__main__":
    main()
