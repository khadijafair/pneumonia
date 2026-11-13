from pathlib import Path

import pandas as pd
import pytest

from pneumonia.data.index import build_index, find_dataset_root, parse_patient
from pneumonia.data.split import (
    audit_original_split,
    check_no_leakage,
    dedupe_exact,
    patient_split,
    split_summary,
)


@pytest.mark.parametrize(
    "name,label,pid,subtype",
    [
        ("person1_bacteria_1.jpeg", "PNEUMONIA", "P-1", "bacteria"),
        ("person1_virus_6.jpeg", "PNEUMONIA", "P-1", "virus"),
        ("person1946_bacteria_4874.jpeg", "PNEUMONIA", "P-1946", "bacteria"),
        ("IM-0115-0001.jpeg", "NORMAL", "N-115", "normal"),
        ("NORMAL2-IM-0115-0001.jpeg", "NORMAL", "N-115", "normal"),
        ("IM-0005-0001-0002.jpeg", "NORMAL", "N-5", "normal"),
    ],
)
def test_parse_patient(name, label, pid, subtype):
    info = parse_patient(name, label)
    assert info.parsed and info.patient_id == pid and info.subtype == subtype


def test_parse_unknown_kept_as_own_group():
    info = parse_patient("weird.jpeg", "PNEUMONIA")
    assert not info.parsed and info.patient_id == "U-weird"


def _make_dataset(base: Path) -> Path:
    """Fake dataset with nested duplicate copy and __MACOSX noise."""
    root = base / "chest_xray"
    k = 0
    for split in ("train", "val", "test"):
        for p in range(1, 41):
            for _j in range(2):
                k += 1
                f = root / split / "PNEUMONIA" / f"person{p}_bacteria_{k}.jpeg"
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(f"img{k}".encode())
            k += 1
            f = root / split / "NORMAL" / f"IM-{p:04d}-{k:04d}.jpeg"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(f"img{k}".encode())
    # one exact duplicate across train/test
    (root / "test" / "NORMAL" / "IM-0999-0001.jpeg").write_bytes(b"img1")
    (root / "train" / "NORMAL" / ".DS_Store").write_bytes(b"x")
    (root / "chest_xray" / "train" / "NORMAL").mkdir(parents=True)
    (root / "chest_xray" / "val").mkdir()
    (root / "chest_xray" / "test").mkdir()
    (base / "__MACOSX" / "chest_xray" / "train").mkdir(parents=True)
    return base


def test_index_and_split_end_to_end(tmp_path):
    root = find_dataset_root(_make_dataset(tmp_path))
    assert root == tmp_path / "chest_xray"
    df = build_index(root)
    assert len(df) == 3 * 40 * 3 + 1
    assert df["parsed"].all()

    audit = audit_original_split(df)
    # same person ids reused in every original split -> leakage is detected
    assert audit["patients_shared_train_test"] == 80
    assert audit["identical_images_train_test"] == 1

    kept, dropped = dedupe_exact(df)
    assert len(dropped) == 1
    out = patient_split(kept, seed=0)
    check_no_leakage(out)
    s = split_summary(out)
    assert set(s.index) == {"train", "val", "test"}
    assert s["images"].sum() == len(kept)
    frac = s["images"] / s["images"].sum()
    assert 0.08 < frac["test"] < 0.25 and 0.08 < frac["val"] < 0.25
    assert (s["pneumonia_rate"] - kept["target"].mean()).abs().max() < 0.1


def test_split_is_deterministic(tmp_path):
    df = build_index(find_dataset_root(_make_dataset(tmp_path)))
    a = patient_split(dedupe_exact(df)[0], seed=7)["split"]
    b = patient_split(dedupe_exact(df)[0], seed=7)["split"]
    pd.testing.assert_series_equal(a, b)


def test_leakage_check_raises():
    df = pd.DataFrame({"patient_id": ["P-1", "P-1"], "md5": ["a", "b"], "split": ["train", "test"]})
    with pytest.raises(AssertionError):
        check_no_leakage(df)
