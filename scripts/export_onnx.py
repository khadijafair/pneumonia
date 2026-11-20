"""Step 5c: export the DenseNet121 to ONNX with the class-activation map computed in the graph.

For this network, Grad-CAM needs no backward pass. The head is
    logit = b + sum_k w_k * mean_ij A_k(i, j)
(global average pooling then one linear layer, A = last ReLU feature maps), so
d logit / d A_k(i, j) = w_k / (H * W): the Grad-CAM weights are the classifier weights, and
    Grad-CAM = ReLU( sum_k w_k * A_k ) / (H * W)    (= CAM, Zhou et al. 2016).
The ONNX model therefore returns both the probability and the 7x7 map in one forward pass,
which lets the demo run entirely in the browser (onnxruntime-web), with no server.

  python scripts/export_onnx.py --model-dir export/densenet121
Writes model.onnx next to model.safetensors and checks it against PyTorch and against Grad-CAM.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
from safetensors.torch import load_file
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))


class ProbAndCAM(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.features = net.features
        self.register_buffer("weight", net.classifier.weight[0].detach().clone())  # (K,)
        self.register_buffer("bias", net.classifier.bias[0].detach().clone())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.features(x)  # (1, K, H, W), after the final ReLU
        cam = torch.einsum("k,bkhw->bhw", self.weight, a)  # (1, H, W)
        logit = cam.mean(dim=(1, 2)) + self.bias
        return torch.sigmoid(logit), torch.relu(cam)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    args = ap.parse_args()

    cfg = json.loads((args.model_dir / "config.json").read_text())
    net = timm.create_model(cfg["timm_name"], pretrained=False, num_classes=1)
    net.load_state_dict(load_file(args.model_dir / "model.safetensors"))
    net.eval()
    wrapped = ProbAndCAM(net).eval()
    s = cfg["img_size"]
    x = torch.randn(1, 3, s, s, generator=torch.Generator().manual_seed(0))

    with torch.no_grad():
        p_torch, cam_torch = wrapped(x)
        assert torch.allclose(p_torch, torch.sigmoid(net(x)[:, 0]), atol=1e-5), "head mismatch"

    out = args.model_dir / "model.onnx"
    torch.onnx.export(
        wrapped,
        (x,),
        out,
        input_names=["image"],
        output_names=["prob", "cam"],
        opset_version=17,
        dynamo=False,
    )

    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(out))
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    p_onnx, cam_onnx = sess.run(None, {"image": x.numpy()})
    assert np.allclose(p_onnx, p_torch.numpy(), atol=1e-4), (p_onnx, p_torch)
    assert np.allclose(cam_onnx, cam_torch.numpy(), atol=1e-3)

    # Same map as the backward-pass Grad-CAM used by the Gradio demo (after normalisation)
    from gradcam import GradCAM

    _logit, heat = GradCAM(net, net.features)(x)
    c = torch.nn.functional.interpolate(
        torch.from_numpy(cam_onnx)[:, None], size=(s, s), mode="bilinear", align_corners=False
    )[0, 0].numpy()
    c = (c - c.min()) / max(c.max() - c.min(), 1e-12)
    max_diff = float(np.abs(c - heat).max())
    assert max_diff < 1e-3, max_diff

    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        sess.run(None, {"image": x.numpy()})
        times.append(time.perf_counter() - t0)
    cfg["onnx"] = {
        "file": "model.onnx",
        "outputs": {"prob": "probability of pneumonia", "cam": "7x7 class-activation map"},
        "cpu_latency_ms_median": round(1000 * float(np.median(times)), 1),
    }
    (args.model_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(
        f"model.onnx: {out.stat().st_size / 1e6:.1f} MB | |p_onnx - p_torch| = "
        f"{abs(float(p_onnx[0]) - float(p_torch[0])):.1e} | Grad-CAM vs CAM max diff = "
        f"{max_diff:.1e} | CPU latency (median of 20) = {cfg['onnx']['cpu_latency_ms_median']} ms"
    )


if __name__ == "__main__":
    main()
