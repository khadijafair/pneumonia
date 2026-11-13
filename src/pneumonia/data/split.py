"""Patient-level, label-stratified train/val/test split and leakage audit."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def dedupe_exact(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop byte-identical images (same MD5), keeping the first. Returns (kept, dropped)."""
    dup_mask = df.duplicated("md5", keep="first")
    return df[~dup_mask].reset_index(drop=True), df[dup_mask].reset_index(drop=True)


def audit_original_split(df: pd.DataFrame) -> dict:
    """Measure leakage in the dataset's original Kaggle split."""
    patients = {s: set(g["patient_id"]) for s, g in df.groupby("orig_split")}
    hashes = {s: set(g["md5"]) for s, g in df.groupby("orig_split")}
    out = {"images_per_split": df["orig_split"].value_counts().to_dict()}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for a, b in pairs:
        if a in patients and b in patients:
            shared = patients[a] & patients[b]
            out[f"patients_shared_{a}_{b}"] = len(shared)
            out[f"images_in_{b}_of_patients_shared_{a}_{b}"] = int(
                df[(df["orig_split"] == b) & df["patient_id"].isin(shared)].shape[0]
            )
            out[f"identical_images_{a}_{b}"] = len(hashes[a] & hashes[b])
    return out


def _one_fold(df: pd.DataFrame, n_splits: int, seed: int) -> tuple[pd.Index, pd.Index]:
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest_idx, held_idx = next(sgkf.split(df, df["target"], groups=df["patient_id"]))
    return df.index[rest_idx], df.index[held_idx]


def patient_split(
    df: pd.DataFrame, test_frac: float = 0.15, val_frac: float = 0.15, seed: int = 42
) -> pd.DataFrame:
    """Assign each row to train/val/test so that no patient appears in two splits.

    Fractions are approximate (whole patients move together); labels stay stratified.
    """
    if not 0 < test_frac < 0.5 or not 0 < val_frac < 0.5:
        raise ValueError("fractions must be in (0, 0.5)")
    df = df.reset_index(drop=True).copy()
    rest, test = _one_fold(df, round(1 / test_frac), seed)
    rest_df = df.loc[rest]
    val_share = val_frac / (1 - test_frac)
    train, val = _one_fold(rest_df, round(1 / val_share), seed)
    df["split"] = "train"
    df.loc[val, "split"] = "val"
    df.loc[test, "split"] = "test"
    check_no_leakage(df)
    return df


def check_no_leakage(df: pd.DataFrame, col: str = "split") -> None:
    per_patient = df.groupby("patient_id")[col].nunique()
    bad = per_patient[per_patient > 1]
    if len(bad):
        examples = list(bad.index[:5])
        raise AssertionError(f"{len(bad)} patients appear in several splits, e.g. {examples}")
    per_hash = df.groupby("md5")[col].nunique()
    if (per_hash > 1).any():
        raise AssertionError("identical images appear in several splits")


def split_summary(df: pd.DataFrame, col: str = "split") -> pd.DataFrame:
    g = df.groupby(col)
    out = pd.DataFrame(
        {
            "images": g.size(),
            "patients": g["patient_id"].nunique(),
            "pneumonia_rate": g["target"].mean().round(4),
            "bacteria": g["subtype"].apply(lambda s: int((s == "bacteria").sum())),
            "virus": g["subtype"].apply(lambda s: int((s == "virus").sum())),
        }
    )
    return out.reindex([s for s in ("train", "val", "test") if s in out.index])
