"""RSNA external set: index, DICOM reading, and the full predict + evaluate path on fake data."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pydicom = pytest.importorskip("pydicom")
from pydicom.dataset import FileDataset, FileMetaDataset  # noqa: E402
from pydicom.uid import ExplicitVRLittleEndian, generate_uid  # noqa: E402

from pneumonia.data.rsna import build_rsna_index, read_dicom, stratified_sample  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def write_dicom(path: Path, arr: np.ndarray, photometric: str = "MONOCHROME2") -> None:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.Rows, ds.Columns = arr.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photometric
    ds.BitsAllocated = ds.BitsStored = arr.dtype.itemsize * 8
    ds.HighBit = ds.BitsStored - 1
    ds.PixelRepresentation = 0
    ds.PixelData = arr.tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(path, enforce_file_format=True)


def make_rsna(base: Path, n_per_class: int = 6) -> Path:
    rng = np.random.default_rng(0)
    comp = base / "rsna-pneumonia-detection-challenge"
    labels, classes = [], []
    k = 0
    for cls, target in (
        ("Lung Opacity", 1),
        ("Normal", 0),
        ("No Lung Opacity / Not Normal", 0),
    ):
        for _ in range(n_per_class):
            pid = f"pid{k:03d}"
            k += 1
            write_dicom(
                comp / "stage_2_train_images" / f"{pid}.dcm",
                rng.integers(0, 255, (48, 40), dtype=np.uint8),
            )
            classes.append({"patientId": pid, "class": cls})
            if target:  # two boxes -> two rows, as in the real file
                for _b in range(2):
                    labels.append(
                        {
                            "patientId": pid,
                            "x": 10.0,
                            "y": 5.0,
                            "width": 8.0,
                            "height": 9.0,
                            "Target": 1,
                        }
                    )
            else:
                labels.append(
                    {
                        "patientId": pid,
                        "x": None,
                        "y": None,
                        "width": None,
                        "height": None,
                        "Target": 0,
                    }
                )
    pd.DataFrame(labels).to_csv(comp / "stage_2_train_labels.csv", index=False)
    pd.DataFrame(classes).to_csv(comp / "stage_2_detailed_class_info.csv", index=False)
    return base


def test_build_index_one_row_per_patient(tmp_path):
    df, root = build_rsna_index(make_rsna(tmp_path))
    assert len(df) == 18 and df["patient_id"].is_unique
    assert df["target"].sum() == 6
    assert set(df["subtype"]) == {"rsna_opacity", "rsna_normal", "rsna_not_normal"}
    opac = df[df.subtype == "rsna_opacity"].iloc[0]
    assert len(json.loads(opac["boxes"])) == 2
    assert (root / df["path"].iloc[0]).exists()
    s = stratified_sample(df, 9)
    assert len(s) == 9 and s["subtype"].value_counts().min() == 3


def test_read_dicom_8bit_16bit_and_inverted(tmp_path):
    a8 = np.arange(0, 200, dtype=np.uint8).reshape(10, 20)
    write_dicom(tmp_path / "a.dcm", a8)
    np.testing.assert_array_equal(np.asarray(read_dicom(tmp_path / "a.dcm"))[..., 0], a8)

    a16 = (np.arange(200, dtype=np.uint16) * 20).reshape(10, 20)
    write_dicom(tmp_path / "b.dcm", a16)
    b = np.asarray(read_dicom(tmp_path / "b.dcm"))[..., 0]
    assert b.dtype == np.uint8 and b.min() == 0 and b.max() == 255

    write_dicom(tmp_path / "c.dcm", a8, photometric="MONOCHROME1")
    c = np.asarray(read_dicom(tmp_path / "c.dcm"))[..., 0]
    np.testing.assert_array_equal(c, 255 - a8)


def test_predict_and_evaluate_external_end_to_end(tmp_path):
    pytest.importorskip("lightning")
    from hydra import compose, initialize_config_dir
    from test_train import _fake_split

    from pneumonia.train import CONFIG_DIR, run

    csv = _fake_split(tmp_path)
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "model.name=resnet10t",
                "model.key=smoke",
                "model.pretrained=false",
                f"data.split_csv={csv}",
                "data.img_size=32",
                "data.batch_size=4",
                "data.num_workers=0",
                "train.epochs=1",
                "train.precision=32-true",
                "train.accelerator=cpu",
                f"out_dir={tmp_path / 'outputs'}",
                f"mlflow.tracking_uri=sqlite:///{tmp_path / 'mlflow.db'}",
                f"mlflow.artifact_location={tmp_path / 'mlartifacts'}",
            ],
        )
    run(cfg)
    rsna = make_rsna(tmp_path / "rsna")
    results = tmp_path / "results_rsna"
    (results / "smoke").mkdir(parents=True)
    for split in ("val", "test"):
        src = tmp_path / "outputs" / "smoke" / f"pred_{split}.csv"
        (results / "smoke" / f"pred_{split}.csv").write_text(src.read_text())

    env = {"PYTHONPATH": str(ROOT / "src"), "MLFLOW_DISABLE_AGENT_HINT": "1"}
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/predict_external.py"),
            "--rsna",
            str(rsna),
            "--ckpt",
            str(tmp_path / "outputs/smoke/best.ckpt"),
            "--results",
            str(results),
            "--img-size",
            "32",
            "--num-workers",
            "0",
        ],
        check=True,
        env={**env, "PATH": "/usr/bin:/bin"},
    )
    pred = pd.read_csv(results / "smoke" / "pred_rsna.csv")
    assert len(pred) == 18 and pred["prob"].between(0, 1).all()

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluate_external.py"),
            "--results",
            str(results),
            "--n-boot",
            "50",
        ],
        check=True,
        env={**env, "PATH": "/usr/bin:/bin"},
    )
    rep = json.loads((results / "evaluation_external.json").read_text())
    assert rep["rsna_counts"]["rsna_opacity"] == 6
    assert 0 <= rep["models"]["smoke"]["rsna_main"]["auroc"] <= 1
