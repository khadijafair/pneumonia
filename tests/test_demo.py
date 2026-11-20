"""Export -> demo loading -> prediction with Grad-CAM, on a randomly initialised DenseNet121."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))

from gradcam import GradCAM, overlay  # noqa: E402


def test_gradcam_shape_range_and_overlay():
    torch.manual_seed(0)
    model = timm.create_model("densenet121", pretrained=False, num_classes=1).eval()
    cam = GradCAM(model, model.features)
    logit, heat = cam(torch.randn(1, 3, 96, 96))
    assert isinstance(logit, float)
    assert heat.shape == (96, 96) and heat.min() >= 0 and heat.max() <= 1
    img = np.zeros((50, 70, 3), dtype=np.uint8)
    out = overlay(img, heat)
    assert out.shape == (50, 70, 3) and out.dtype == np.uint8


def test_export_then_demo_predict(tmp_path):
    from PIL import Image

    torch.manual_seed(0)
    net = timm.create_model("densenet121", pretrained=False, num_classes=1)
    ckpt = {
        "state_dict": {f"net.{k}": v for k, v in net.state_dict().items()}
        | {"pos_weight": torch.tensor([0.37])},
        "hyper_parameters": {"name": "densenet121", "pretrained": True},
    }
    torch.save(ckpt, tmp_path / "densenet121.ckpt")
    pd.DataFrame({"target": [0, 0, 1, 1, 1, 1], "prob": [0.1, 0.4, 0.3, 0.6, 0.8, 0.9]}).to_csv(
        tmp_path / "pred_val.csv", index=False
    )
    out = tmp_path / "export"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/export_model.py"),
            "--ckpt",
            str(tmp_path / "densenet121.ckpt"),
            "--pred-val",
            str(tmp_path / "pred_val.csv"),
            "--out",
            str(out),
            "--img-size",
            "64",
        ],
        check=True,
    )
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["threshold"] == 0.3  # lowest positive kept: sensitivity 100 % >= 98 %
    assert (out / "model.safetensors").stat().st_size > 1e6

    import app

    probs, heat, text = app.predict(Image.new("L", (80, 100), 128), path=str(out))
    assert 0 <= probs["pneumonie"] <= 1 and abs(sum(probs.values()) - 1) < 1e-6
    assert heat.ndim == 3 and heat.shape[1] == 512
    assert "seuil 0.30" in text
    # the exported weights reproduce the original network exactly
    model, tf, _cam, _cfg = app.load(str(out))
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(model(x), net.eval()(x), atol=1e-6)


def test_export_onnx_matches_torch_and_gradcam(tmp_path):
    pytest.importorskip("onnxruntime")
    from safetensors.torch import save_file

    torch.manual_seed(0)
    net = timm.create_model("densenet121", pretrained=False, num_classes=1)
    save_file(
        {k: v.contiguous() for k, v in net.state_dict().items()}, tmp_path / "model.safetensors"
    )
    cfg = {
        "timm_name": "densenet121",
        "img_size": 64,
        "mean": [0.5] * 3,
        "std": [0.25] * 3,
        "threshold": 0.5,
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    # the script itself asserts ONNX == PyTorch and CAM == backward-pass Grad-CAM
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/export_onnx.py"), "--model-dir", str(tmp_path)],
        check=True,
    )
    assert (tmp_path / "model.onnx").stat().st_size > 1e6
    assert json.loads((tmp_path / "config.json").read_text())["onnx"]["cpu_latency_ms_median"] > 0
