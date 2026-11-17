"""Check that "the same patient id in train and test" really means the same person.

The Kaggle file names give patient numbers, but nothing guarantees that numbering is shared
between the train/ and test/ folders. We compare image similarity of three kinds of pairs:
  A. same id, train image vs test image        (the claim to verify)
  B. same id, two images inside train          (reference: surely the same person)
  C. different patients, same label, train vs test (reference: different people)
If A looks like B and clearly above C, the shared ids are real patients -> real leakage.

The measure must first separate B from C; otherwise the test is inconclusive.
  --features pixels : correlation of 128x128 grayscale pixels (fast, weak)
  --features deep   : cosine similarity of CNN embeddings (ImageNet DenseNet121, or the
                      backbone of a trained checkpoint with --ckpt)

Usage: python scripts/audit_patient_ids.py --features deep [--ckpt outputs/densenet121/best.ckpt]
Output: data/splits/patient_id_audit[_deep].md
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
    # pixels: vectors are standardized -> Pearson correlation; deep: L2-normalized -> cosine
    return float(np.dot(a, b) / a.size) if a.size == SIZE * SIZE else float(np.dot(a, b))


def deep_featurizer(ckpt: Path | None):
    import timm
    import torch
    from PIL import Image as _Image
    from torchvision import transforms as T

    if ckpt:
        from pneumonia.model import PneumoniaClassifier

        net = PneumoniaClassifier.load_from_checkpoint(ckpt, map_location="cpu").net
    else:
        net = timm.create_model("densenet121.tv_in1k", pretrained=True, num_classes=0)
    cfg = net.pretrained_cfg
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(cfg["mean"], cfg["std"])])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.eval().to(device)

    @torch.no_grad()
    def featurize(path: Path) -> np.ndarray:
        x = tf(_Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        f = net.forward_head(net.forward_features(x), pre_logits=True).flatten()
        f = torch.nn.functional.normalize(f.float(), dim=0)
        return f.cpu().numpy()

    return featurize


def prefix(path: str) -> str:
    return "NORMAL2-IM" if Path(path).name.upper().startswith("NORMAL2-") else "IM"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=Path("data/splits/split.csv"))
    ap.add_argument("--root", type=Path, default=None, help="folder containing chest_xray/")
    ap.add_argument("--features", choices=["pixels", "deep"], default="pixels")
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.out is None:
        suffix = "" if args.features == "pixels" else "_deep" + ("_ckpt" if args.ckpt else "")
        args.out = Path(f"data/splits/patient_id_audit{suffix}.md")

    from pneumonia.data.index import find_dataset_root

    df = pd.read_csv(args.index)
    df = df[df["parsed"]].drop_duplicates("md5").reset_index(drop=True)
    root = find_dataset_root(args.root or Path(df["root"].iloc[0]))
    rng = np.random.default_rng(args.seed)
    cache: dict[str, np.ndarray] = {}
    featurize = load if args.features == "pixels" else deep_featurizer(args.ckpt)

    def vec(p: str) -> np.ndarray:
        if p not in cache:
            cache[p] = featurize(root / p)
        return cache[p]

    train = df[df["orig_split"] == "train"]
    test = df[df["orig_split"] == "test"]
    shared = sorted(set(train["patient_id"]) & set(test["patient_id"]))

    lines = [f"# Patient id audit ({args.features} features), original train vs test", ""]
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
                        "median_similarity": round(float(np.median(vals)), 4),
                        "mean": round(float(np.mean(vals)), 4),
                    }
                )
        if b and c:
            p_bc = mannwhitneyu(b, c, alternative="greater").pvalue
            ok = "the measure separates people" if p_bc < 0.01 else "INCONCLUSIVE measure"
            tests.append(f"- {label}: sanity B > C, Mann-Whitney p = {p_bc:.1e} -> {ok}")
        if a and c:
            p = mannwhitneyu(a, c, alternative="greater").pvalue
            win = float(np.mean(np.array(a) > np.array(c)))
            tests.append(f"- {label}: A > C for {win:.1%} of pairs, Mann-Whitney p = {p:.1e}")

    measure = (
        "Pearson correlation, 128x128 grayscale"
        if args.features == "pixels"
        else f"cosine of embeddings ({args.ckpt or 'ImageNet DenseNet121'})"
    )
    lines += ["", f"## Image similarity ({measure})", ""]
    lines.append(pd.DataFrame(results).to_markdown(index=False))
    lines += ["", *tests]
    lines += [
        "",
        "Reading: first B must be clearly above C (else the measure cannot tell people apart).",
        "Then: A close to B and above C = shared ids are the same people (real leakage);",
        "A close to C = the numbering is not shared between folders.",
        "",
    ]
    args.out.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
