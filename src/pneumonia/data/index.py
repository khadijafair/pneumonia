"""Index the Kermany chest X-ray dataset (Kaggle: paultimothymooney/chest-xray-pneumonia).

Each image gets: path, label, subtype, original split, patient id and an MD5 content hash.
Patient ids come from the file names:
  - PNEUMONIA : person<N>_bacteria_<k>.jpeg / person<N>_virus_<k>.jpeg  -> "P-<N>"
  - NORMAL    : IM-<N>-<k>.jpeg, NORMAL2-IM-<N>-<k>.jpeg, IM-<N>-<k>-<j>.jpeg -> "N-<N>"
The NORMAL2- prefix is merged with IM-<N> on purpose: if they are different people the
groups are only coarser, which can never create leakage between splits.
Files that match no pattern are kept as their own group and counted in the report.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

LABELS = ("NORMAL", "PNEUMONIA")
SPLITS = ("train", "val", "test")
IMAGE_EXT = {".jpeg", ".jpg", ".png"}

_PNEUMONIA_RE = re.compile(r"^person(\d+)_(bacteria|virus)_\d+$", re.IGNORECASE)
_NORMAL_RE = re.compile(r"^(?:NORMAL2-)?IM-(\d+)-\d+(?:-\d+)?$", re.IGNORECASE)


@dataclass(frozen=True)
class PatientInfo:
    patient_id: str
    subtype: str  # "normal", "bacteria", "virus" or "unknown"
    parsed: bool


def parse_patient(filename: str, label: str) -> PatientInfo:
    """Extract the patient id and pneumonia subtype from a file name."""
    stem = Path(filename).stem
    if label == "PNEUMONIA":
        m = _PNEUMONIA_RE.match(stem)
        if m:
            return PatientInfo(f"P-{int(m.group(1))}", m.group(2).lower(), True)
        return PatientInfo(f"U-{stem}", "unknown", False)
    m = _NORMAL_RE.match(stem)
    if m:
        return PatientInfo(f"N-{int(m.group(1))}", "normal", True)
    return PatientInfo(f"U-{stem}", "normal", False)


def md5_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def find_dataset_root(base: Path) -> Path:
    """Return the folder holding train/ val/ test/ (handles chest_xray/chest_xray)."""
    candidates = []
    for p in [base, *base.rglob("*")]:
        if p.is_dir() and "__MACOSX" not in p.parts and all((p / s).is_dir() for s in SPLITS):
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError(f"No folder with train/ val/ test/ under {base}")
    # shortest path = outermost copy; nested duplicates are ignored
    return min(candidates, key=lambda p: len(p.parts))


def build_index(root: Path) -> pd.DataFrame:
    """Scan root/{train,val,test}/{NORMAL,PNEUMONIA}/ and return one row per image."""
    rows = []
    for split in SPLITS:
        for label in LABELS:
            folder = root / split / label
            if not folder.is_dir():
                continue
            for img in sorted(folder.iterdir()):
                if img.name.startswith(".") or img.suffix.lower() not in IMAGE_EXT:
                    continue
                info = parse_patient(img.name, label)
                rows.append(
                    {
                        "path": str(img.relative_to(root)),
                        "label": label,
                        "target": int(label == "PNEUMONIA"),
                        "subtype": info.subtype,
                        "orig_split": split,
                        "patient_id": info.patient_id,
                        "parsed": info.parsed,
                        "md5": md5_file(img),
                    }
                )
    if not rows:
        raise FileNotFoundError(f"No images found under {root}")
    return pd.DataFrame(rows)
