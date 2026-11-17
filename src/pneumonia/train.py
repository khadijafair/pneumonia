"""Train one backbone, evaluate the best checkpoint, save predictions, log everything to MLflow.

python -m pneumonia.train                                  # default model
python -m pneumonia.train -m model=efficientnet_b0,densenet121,vit_small   # compare 3
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf

from pneumonia.data.dataset import ChestXrayDataModule
from pneumonia.model import PneumoniaClassifier, backbone_norm, create_backbone

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _md5(path: str) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def run(cfg: DictConfig) -> dict:
    L.seed_everything(cfg.train.seed, workers=True)
    torch.set_float32_matmul_precision("high")

    mean, std = backbone_norm(create_backbone(cfg.model.name, pretrained=False))
    dm = ChestXrayDataModule(
        cfg.data.split_csv,
        cfg.data.root,
        cfg.data.img_size,
        cfg.data.batch_size,
        cfg.data.num_workers,
        mean,
        std,
    )
    model = PneumoniaClassifier(
        cfg.model.name,
        pretrained=cfg.model.pretrained,
        lr=cfg.model.lr,
        weight_decay=cfg.train.weight_decay,
        pos_weight=dm.pos_weight(),
        epochs=cfg.train.epochs,
    )

    out = Path(cfg.out_dir) / cfg.model.key
    out.mkdir(parents=True, exist_ok=True)
    logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment,
        tracking_uri=cfg.mlflow.tracking_uri,
        artifact_location=str(Path(cfg.mlflow.artifact_location).resolve()),
        run_name=cfg.model.key,
    )
    logger.log_hyperparams(
        {
            **OmegaConf.to_container(cfg, resolve=True),
            "git_commit": _git_commit(),
            "split_csv_md5": _md5(cfg.data.split_csv),
            "pos_weight": round(dm.pos_weight(), 4),
            "n_params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        }
    )

    ckpt = ModelCheckpoint(
        dirpath=out, filename="best", monitor="val_auroc", mode="max", save_top_k=1
    )
    trainer = L.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator=cfg.train.accelerator,
        devices=1,
        precision=cfg.train.precision,
        logger=logger,
        callbacks=[ckpt, EarlyStopping("val_auroc", mode="max", patience=cfg.train.patience)],
        deterministic=False,
        log_every_n_steps=10,
        **OmegaConf.to_container(cfg.trainer_extra, resolve=True),
    )
    trainer.fit(model, datamodule=dm)
    best = ckpt.best_model_path or None
    test_metrics = trainer.test(model, datamodule=dm, ckpt_path=best)[0]

    # Predictions of the best checkpoint on val (to choose the threshold) and test
    for split in ("val", "test"):
        batches = trainer.predict(model, dataloaders=dm.predict_loader(split), ckpt_path=best)
        idx = torch.cat([b[0] for b in batches]).numpy()
        prob = torch.cat([b[1] for b in batches]).float().numpy()
        frame = dm.frame(split).iloc[idx][["path", "patient_id", "label", "target", "subtype"]]
        frame = frame.assign(prob=prob).sort_values("path")
        frame.to_csv(out / f"pred_{split}.csv", index=False)
        logger.experiment.log_artifact(logger.run_id, str(out / f"pred_{split}.csv"))

    summary = {
        "model": cfg.model.key,
        "best_epoch_val_auroc": float(ckpt.best_model_score or float("nan")),
        **{k: round(float(v), 4) for k, v in test_metrics.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.experiment.log_artifact(logger.run_id, str(out / "summary.json"))
    print(json.dumps(summary, indent=2))
    return summary


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="train")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
