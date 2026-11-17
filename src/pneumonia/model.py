"""Lightning module: any timm backbone with a single-logit head."""

from __future__ import annotations

import lightning as L
import timm
import torch
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryRecall,
    BinarySpecificity,
)


def create_backbone(name: str, pretrained: bool) -> nn.Module:
    return timm.create_model(name, pretrained=pretrained, num_classes=1)


def backbone_norm(model: nn.Module) -> tuple[tuple, tuple]:
    cfg = getattr(model, "pretrained_cfg", {}) or {}
    return tuple(cfg.get("mean", (0.485, 0.456, 0.406))), tuple(
        cfg.get("std", (0.229, 0.224, 0.225))
    )


def _metrics(prefix: str) -> MetricCollection:
    return MetricCollection(
        {
            "auroc": BinaryAUROC(),
            "recall": BinaryRecall(),
            "specificity": BinarySpecificity(),
            "accuracy": BinaryAccuracy(),
        },
        prefix=prefix,
    )


class PneumoniaClassifier(L.LightningModule):
    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        pos_weight: float = 1.0,
        epochs: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = create_backbone(name, pretrained)
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))
        self.val_metrics = _metrics("val_")
        self.test_metrics = _metrics("test_")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)

    def _loss(self, logits, y):
        return nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pos_weight)

    def training_step(self, batch, _):
        x, y, _idx = batch
        loss = self._loss(self(x), y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def _eval_step(self, batch, metrics, stage: str):
        x, y, _idx = batch
        logits = self(x)
        self.log(f"{stage}_loss", self._loss(logits, y), on_epoch=True, prog_bar=True)
        metrics.update(torch.sigmoid(logits), y.long())

    def validation_step(self, batch, _):
        self._eval_step(batch, self.val_metrics, "val")

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), prog_bar=True)
        self.val_metrics.reset()

    def test_step(self, batch, _):
        self._eval_step(batch, self.test_metrics, "test")

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def predict_step(self, batch, _):
        x, _y, idx = batch
        return idx, torch.sigmoid(self(x))

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.epochs)
        return {"optimizer": opt, "lr_scheduler": sched}
