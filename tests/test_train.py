"""Smoke test of the full training pipeline on a tiny synthetic dataset (CPU, random weights)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")
from hydra import compose, initialize_config_dir  # noqa: E402
from PIL import Image  # noqa: E402

from pneumonia.train import CONFIG_DIR, run  # noqa: E402


def _fake_split(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    root = tmp_path / "chest_xray"
    rows = []
    for split in ("train", "val", "test"):
        (root / split).mkdir(parents=True)  # find_dataset_root needs train/ val/ test/
        for i in range(8):
            target = int(i % 4 != 0)  # 2 normal / 6 pneumonia: pos_weight = 1/3, not round
            label = "PNEUMONIA" if target else "NORMAL"
            rel = f"{split}/{label}/img_{i}.jpeg"
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 255, (40, 50), dtype=np.uint8)).save(root / rel)
            rows.append(
                {
                    "root": str(root),
                    "path": rel,
                    "label": label,
                    "target": target,
                    "subtype": "bacteria" if target else "normal",
                    "patient_id": f"{split}-{i}",
                    "split": split,
                }
            )
    csv = tmp_path / "split.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def test_training_pipeline_end_to_end(tmp_path):
    csv = _fake_split(tmp_path)
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "model.name=resnet10t",
                "model.key=smoke",
                "model.pretrained=false",
                f"data.split_csv={csv}",
                "data.img_size=32",
                "data.batch_size=4",
                "data.num_workers=0",
                "train.epochs=2",
                "train.precision=32-true",
                "train.accelerator=cpu",
                f"out_dir={tmp_path / 'outputs'}",
                f"mlflow.tracking_uri=sqlite:///{tmp_path / 'mlflow.db'}",
                f"mlflow.artifact_location={tmp_path / 'mlartifacts'}",
            ],
        )
    summary = run(cfg)
    out = tmp_path / "outputs" / "smoke"
    assert (out / "best.ckpt").exists()
    for split in ("val", "test"):
        pred = pd.read_csv(out / f"pred_{split}.csv")
        assert len(pred) == 8 and pred["prob"].between(0, 1).all()
        assert pred["path"].str.startswith(split).all()
    assert "test_auroc" in summary
    assert json.loads((out / "summary.json").read_text())["model"] == "smoke"
    assert any((tmp_path / "mlartifacts").rglob("pred_test.csv"))
