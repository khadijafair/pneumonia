"""Check that "the same patient id in train and test" really means the same person.

The Kaggle file names give patient numbers, but nothing guarantees that numbering is shared
between the train/ and test/ folders. We compare image similarity of three kinds of pairs:
  A. same id, train image vs test image        (the claim to verify)
  B. same id, two images inside train          (reference: surely the same person)
  C. different patients, same label, train vs test (reference: different people)
If A looks like B and clearly above C, the shared ids are real patients -> real leakage.

Usage: python scripts/audit_patient_ids.py [--index data/splits/index.csv]
Output: data/splits/patient_id_audit.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SIZE = 128


def load(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L").resize((SIZE, SIZE), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32).ravel()
    return (a - a.mean()) / (a.std() + 1e-6)


def sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / a.size)  # Pearson correlation of standardized pixels


def prefix(path: str) -> str:
    return "NORMAL2-IM" if Path(path).name.upper().startswith("NORMAL2-") else "IM"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("data/splits/index.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/splits/patient_id_audit.md"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.index)
    df = df[df["parsed"]].drop_duplicates("md5").reset_index(drop=True)
    root = Path(df["root"].iloc[0])
    rng = np.random.default_rng(args.seed)
    cache: dict[str, np.ndarray] = {}

    def vec(p: str) -> np.ndarray:
        if p not in cache:
            cache[p] = load(root / p)
        return cache[p]

    train = df[df["orig_split"] == "train"]
    test = df[df["orig_split"] == "test"]
    shared = sorted(set(train["patient_id"]) & set(test["patient_id"]))

    lines = ["# Patient id audit (train vs test of the original Kaggle split)", ""]
    lines.append(f"Shared patient ids: {len(shared)}")
    sh = df[df["patient_id"].isin(shared)]
    lines.append("")
    lines.append(
        "By label: "
        + ", ".join(f"{k}: {v}" for k, v in sh.groupby("label")["patient_id"].nunique().items())
    )
    normal_sh = sh[sh["label"] == "NORMAL"].copy()
    if len(normal_sh):
        normal_sh["prefix"] = normal_sh["path"].map(prefix)
        combo = (
            normal_sh.groupby("patient_id")
            .apply(
                lambda g: " / ".join(
                    f"{s}:{','.join(sorted(set(g[g.orig_split == s].prefix)))}"
                    for s in ("train", "test")
                ),
                include_groups=False,
            )
            .value_counts()
        )
        lines += ["", "NORMAL shared ids by file-name prefix (train / test):", ""]
        lines += [f"- {k}: {v}" for k, v in combo.items()]

    results = []
    tests: list[str] = []
    for label in ("NORMAL", "PNEUMONIA"):
        tr_l, te_l = train[train.label == label], test[test.label == label]
        ids = [p for p in shared if p in set(tr_l.patient_id) and p in set(te_l.patient_id)]
        a, c = [], []
        for pid in ids:
            p_tr = rng.choice(tr_l.loc[tr_l.patient_id == pid, "path"].to_numpy())
            p_te = rng.choice(te_l.loc[te_l.patient_id == pid, "path"].to_numpy())
            a.append(sim(vec(p_tr), vec(p_te)))
            other = tr_l[tr_l.patient_id != pid]
            c.append(sim(vec(rng.choice(other["path"].to_numpy())), vec(p_te)))
        multi = tr_l.groupby("patient_id").filter(lambda g: len(g) >= 2)
        b = []
        for _pid, g in list(multi.groupby("patient_id"))[:500]:
            p1, p2 = rng.choice(g["path"].to_numpy(), size=2, replace=False)
            b.append(sim(vec(p1), vec(p2)))
        for name, vals in (
            ("A same id train-test", a),
            ("B same id inside train", b),
            ("C different patients train-test", c),
        ):
            if vals:
                results.append(
                    {
                        "label": label,
                        "pairs": name,
                        "n": len(vals),
                        "median_similarity": round(float(np.median(vals)), 3),
                        "mean": round(float(np.mean(vals)), 3),
                    }
                )
        if a and c:
            p = mannwhitneyu(a, c, alternative="greater").pvalue
            win = float(np.mean(np.array(a) > np.array(c)))
            tests.append(f"- {label}: A > C for {win:.1%} of pairs, Mann-Whitney p = {p:.1e}")

    lines += ["", "## Image similarity (Pearson correlation, 128x128 grayscale)", ""]
    lines.append(pd.DataFrame(results).to_markdown(index=False))
    lines += ["", *tests]
    lines += [
        "",
        "Reading: if A is close to B and clearly above C, the shared ids are the same people",
        "(real leakage). If A is close to C, the numbering is not shared between folders.",
        "",
    ]
    args.out.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
