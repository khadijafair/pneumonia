"""Step 5b: publish the model and the Gradio demo on Hugging Face (free).

Needs a Hugging Face *write* token in the environment variable HF_TOKEN.
  python scripts/publish_hf.py --model-dir export/densenet121 --examples-dir demo/examples
Creates (or updates) <user>/pneumonia-x-densenet121 (model) and <user>/pneumonia-x (Space).
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODEL_CARD = """---
license: mit
library_name: timm
tags: [image-classification, medical-imaging, chest-x-ray, pneumonia, grad-cam]
pipeline_tag: image-classification
---
# Pneumonia-X — DenseNet121 (chest X-ray, pneumonia vs normal)

**Educational model, not a medical device.**

- Trained on the Kermany pediatric chest X-ray dataset (children 1–5 years, one hospital), with a
  **patient-level** train/val/test split. Code, data audit and full evaluation:
  https://github.com/khadijafair/pneumonia
- Operating threshold: `{threshold}` ({threshold_rule}).
{metrics}
- Known limits (measured): on adults the specificity drops at the pediatric threshold, and the
  model flags most X-rays with *other* abnormalities: it detects "abnormal" rather than
  pneumonia specifically.

Files: `model.safetensors` (timm `{timm_name}`, 1 output logit), `config.json` (preprocessing,
threshold, metrics).
"""

SPACE_README = """---
title: Pneumonia-X
emoji: 🫁
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: {gradio_version}
app_file: app.py
pinned: false
license: mit
short_description: Pneumonia on chest X-rays, Grad-CAM, external validation
---
Demo of [Pneumonia-X](https://github.com/khadijafair/pneumonia). Educational, not a medical device.
"""

SPACE_REQUIREMENTS = """--extra-index-url https://download.pytorch.org/whl/cpu
torch
torchvision
timm>=1.0
safetensors
huggingface_hub
matplotlib
"""


def main() -> None:
    import gradio
    from huggingface_hub import HfApi

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--examples-dir", type=Path, default=ROOT / "demo" / "examples")
    ap.add_argument("--model-name", default="pneumonia-x-densenet121")
    ap.add_argument("--space-name", default="pneumonia-x")
    args = ap.parse_args()

    api = HfApi()  # reads HF_TOKEN from the environment
    user = api.whoami()["name"]
    model_repo, space_repo = f"{user}/{args.model_name}", f"{user}/{args.space_name}"

    cfg = json.loads((args.model_dir / "config.json").read_text())
    metrics = "\n".join(f"- {k}: `{v}`" for k, v in cfg.get("metrics", {}).items())
    with tempfile.TemporaryDirectory() as tmp:
        m = Path(tmp) / "model"
        shutil.copytree(args.model_dir, m)
        (m / "README.md").write_text(MODEL_CARD.format(**{**cfg, "metrics": metrics}))
        api.create_repo(model_repo, repo_type="model", exist_ok=True)
        api.upload_folder(repo_id=model_repo, folder_path=m, commit_message="Upload model")

        s = Path(tmp) / "space"
        s.mkdir()
        for f in ("app.py", "gradcam.py"):
            shutil.copy(ROOT / "demo" / f, s / f)
        if args.examples_dir.exists():
            shutil.copytree(args.examples_dir, s / "examples")
        (s / "model_repo.txt").write_text(model_repo + "\n")
        (s / "requirements.txt").write_text(SPACE_REQUIREMENTS)
        (s / "README.md").write_text(SPACE_README.format(gradio_version=gradio.__version__))
        api.create_repo(space_repo, repo_type="space", space_sdk="gradio", exist_ok=True)
        api.upload_folder(
            repo_id=space_repo, repo_type="space", folder_path=s, commit_message="Upload demo"
        )

    print(f"Model : https://huggingface.co/{model_repo}")
    print(f"Space : https://huggingface.co/spaces/{space_repo}")


if __name__ == "__main__":
    main()
