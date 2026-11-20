"""PyTorch dataset and Lightning data module built on data/splits/split.csv."""

from __future__ import annotations

from pathlib import Path

import lightning as L
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from pneumonia.data.index import find_dataset_root


def load_image(path: Path) -> Image.Image:
    if path.suffix.lower() == ".dcm":
        from pneumonia.data.rsna import read_dicom

        return read_dicom(path)
    return Image.open(path).convert("RGB")


def build_transforms(img_size: int, mean, std, train: bool) -> T.Compose:
    # No horizontal flip: on a chest X-ray the heart side is meaningful.
    if train:
        aug = [
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            T.RandomRotation(10),
            T.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    else:
        aug = [T.Resize((img_size, img_size))]
    return T.Compose([*aug, T.ToTensor(), T.Normalize(mean, std)])


class ChestXrayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: Path, transform: T.Compose):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = load_image(self.root / row["path"])
        return self.transform(img), torch.tensor(float(row["target"])), i


class ChestXrayDataModule(L.LightningDataModule):
    def __init__(
        self,
        split_csv: str,
        root: str | None,
        img_size: int,
        batch_size: int,
        num_workers: int,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.df = pd.read_csv(split_csv)
        base = Path(root) if root else Path(self.df["root"].iloc[0])
        self.root = find_dataset_root(base)
        self.img_size, self.batch_size, self.num_workers = img_size, batch_size, num_workers
        self.mean, self.std = mean, std

    def frame(self, split: str) -> pd.DataFrame:
        return self.df[self.df["split"] == split].reset_index(drop=True)

    def pos_weight(self) -> float:
        tr = self.frame("train")["target"]
        return float((tr == 0).sum() / max((tr == 1).sum(), 1))

    def _loader(self, split: str, train: bool) -> DataLoader:
        tf = build_transforms(self.img_size, self.mean, self.std, train)
        return DataLoader(
            ChestXrayDataset(self.frame(split), self.root, tf),
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self):
        return self._loader("train", True)

    def val_dataloader(self):
        return self._loader("val", False)

    def test_dataloader(self):
        return self._loader("test", False)

    def predict_loader(self, split: str) -> DataLoader:
        return self._loader(split, False)
