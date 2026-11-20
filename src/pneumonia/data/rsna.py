"""External test set: RSNA Pneumonia Detection Challenge (Kaggle competition, adults, DICOM).

Files used (searched anywhere under the given folder):
  stage_2_train_labels.csv        patientId, x, y, width, height, Target (one row per box)
  stage_2_detailed_class_info.csv patientId, class
  stage_2_train_images/<patientId>.dcm
Classes: "Lung Opacity" (pneumonia, Target = 1, with boxes), "Normal",
"No Lung Opacity / Not Normal" (other abnormality, Target = 0).
One image per patient in this set, so image-level and patient-level resampling coincide.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

SUBTYPES = {
    "Lung Opacity": "rsna_opacity",
    "Normal": "rsna_normal",
    "No Lung Opacity / Not Normal": "rsna_not_normal",
}


def _find(base: Path, name: str) -> Path:
    hits = sorted(base.rglob(name), key=lambda p: len(p.parts))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {base}")
    return hits[0]


def build_rsna_index(base: Path) -> tuple[pd.DataFrame, Path]:
    """Return (one row per image, root folder that `path` is relative to)."""
    labels = pd.read_csv(_find(base, "stage_2_train_labels.csv"))
    classes = pd.read_csv(_find(base, "stage_2_detailed_class_info.csv")).drop_duplicates()
    img_dir = _find(base, "stage_2_train_images")
    root = img_dir.parent

    boxes = (
        labels.dropna(subset=["x"])
        .groupby("patientId")[["x", "y", "width", "height"]]
        .apply(lambda g: json.dumps(g.round(1).to_numpy().tolist()), include_groups=False)
        .rename("boxes")
    )
    df = (
        labels.groupby("patientId")["Target"]
        .max()
        .rename("target")
        .reset_index()
        .merge(classes.rename(columns={"class": "rsna_class"}), on="patientId", how="left")
        .merge(boxes, on="patientId", how="left")
    )
    if df["patientId"].duplicated().any():
        raise ValueError("a patient has several classes in stage_2_detailed_class_info.csv")
    df["boxes"] = df["boxes"].fillna("[]")
    df["path"] = img_dir.name + "/" + df["patientId"] + ".dcm"
    df["patient_id"] = "R-" + df["patientId"]
    df["label"] = np.where(df["target"] == 1, "PNEUMONIA", "NORMAL")
    df["subtype"] = df["rsna_class"].map(SUBTYPES).fillna("rsna_unknown")
    return df[["path", "patient_id", "label", "target", "subtype", "boxes"]], root


def stratified_sample(df: pd.DataFrame, n: int | None, seed: int = 0) -> pd.DataFrame:
    """Keep the class mix (by subtype) while limiting the number of images."""
    if not n or n >= len(df):
        return df.reset_index(drop=True)
    frac = n / len(df)
    parts = [
        g.sample(max(1, round(len(g) * frac)), random_state=seed)
        for _, g in df.groupby("subtype", sort=True)
    ]
    return pd.concat(parts).reset_index(drop=True)


def read_dicom(path: Path) -> Image.Image:
    """DICOM -> 8-bit RGB PIL image, as close as possible to the JPEGs used in training.

    8-bit images (the RSNA case) are kept as they are; deeper images are rescaled min-max.
    MONOCHROME1 (inverted grayscale) is flipped back.
    """
    import pydicom

    ds = pydicom.dcmread(path)
    a = ds.pixel_array
    if a.dtype != np.uint8:
        a = a.astype(np.float32)
        a = (a - a.min()) / max(float(a.max() - a.min()), 1e-6) * 255
        a = a.astype(np.uint8)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        a = 255 - a
    return Image.fromarray(a).convert("RGB")
