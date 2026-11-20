"""Grad-CAM for a single-logit classifier (Selvaraju et al., 2017), written without extra libraries.

Heat map = ReLU( sum_k  w_k * A_k ), with A_k the k-th feature map of the target layer and
w_k the spatial mean of d(logit)/d(A_k). Red areas pushed the "pneumonia" score up.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self._acts: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None
        target_layer.register_forward_hook(self._save)

    def _save(self, _module, _inp, out: torch.Tensor) -> None:
        self._acts = out
        if out.requires_grad:  # plain inference (torch.no_grad) must keep working
            out.register_hook(lambda g: setattr(self, "_grads", g))

    def __call__(self, x: torch.Tensor) -> tuple[float, np.ndarray]:
        """x: (1, 3, H, W). Returns (logit, heat map in [0, 1] of shape (H, W))."""
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            x = x.requires_grad_(False)
            logit = self.model(x)[0, 0]
            logit.backward()
        weights = self._grads.mean(dim=(2, 3), keepdim=True)  # (1, K, 1, 1)
        cam = F.relu((weights * self._acts).sum(dim=1, keepdim=True)).detach()
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / cam.max() if cam.max() > 0 else cam
        return float(logit.detach()), cam.numpy()


def overlay(image: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """image: (H, W, 3) uint8, cam: (h, w) in [0, 1] -> (H, W, 3) uint8 with a jet heat map."""
    from matplotlib import colormaps
    from PIL import Image

    h, w = image.shape[:2]
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    heat = colormaps["jet"](np.asarray(cam_img) / 255.0)[..., :3] * 255
    return ((1 - alpha) * image + alpha * heat).clip(0, 255).astype(np.uint8)
