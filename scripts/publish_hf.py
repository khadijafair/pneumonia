"""Step 5b: publish the model and the demo on Hugging Face.

Needs a Hugging Face *write* token in the environment variable HF_TOKEN.
  python scripts/publish_hf.py --model-dir export/densenet121 --examples-dir demo/examples
Creates (or updates) <user>/pneumonia-x-densenet121 (model) and <user>/pneumonia-x (Space).

Default: a *static* Space (free for everyone): HTML + JavaScript running model.onnx in the
visitor's browser (run scripts/export_onnx.py first). --gradio publishes the Python/Gradio app
instead, which Hugging Face only hosts on its free CPU for PRO accounts.
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

Files: `model.safetensors` (timm `{timm_name}`, 1 output logit), `model.onnx` (outputs the
probability and the 7x7 class-activation map, equal to Grad-CAM for this architecture),
`config.json` (preprocessing, threshold, metrics, ONNX CPU latency).
"""

STATIC_README = """---
title: Pneumonia-X
emoji: 🫁
colorFrom: blue
colorTo: red
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Pneumonia on chest X-rays, Grad-CAM, runs in the browser
---
Demo of [Pneumonia-X](https://github.com/khadijafair/pneumonia): DenseNet121 exported to ONNX,
run in the browser with ONNX Runtime Web (no server). Educational, not a medical device.
"""

EXAMPLE_LABELS = {
    "normal": "Normale",
    "pneumonie_bacterienne": "Pneumonie bactérienne",
    "pneumonie_virale": "Pneumonie virale",
}

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
    from huggingface_hub import HfApi

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--examples-dir", type=Path, default=ROOT / "demo" / "examples")
    ap.add_argument("--model-name", default="pneumonia-x-densenet121")
    ap.add_argument("--space-name", default="pneumonia-x")
    ap.add_argument("--gradio", action="store_true", help="Gradio Space (PRO) instead of static")
    args = ap.parse_args()
    if not args.gradio and not (args.model_dir / "model.onnx").exists():
        raise SystemExit("model.onnx missing: run scripts/export_onnx.py first")

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
        if args.examples_dir.exists():
            shutil.copytree(args.examples_dir, s / "examples")
            for f in (s / "examples").glob(".*"):
                f.unlink()
        if args.gradio:
            for f in ("app.py", "gradcam.py"):
                shutil.copy(ROOT / "demo" / f, s / f)
            (s / "model_repo.txt").write_text(model_repo + "\n")
            (s / "requirements.txt").write_text(SPACE_REQUIREMENTS)
            import gradio

            (s / "README.md").write_text(SPACE_README.format(gradio_version=gradio.__version__))
            sdk = "gradio"
        else:
            for f in ("index.html", "pneumonia.js"):
                shutil.copy(ROOT / "demo_static" / f, s / f)
            examples = (
                [
                    {"file": f.name, "label": EXAMPLE_LABELS.get(f.stem, f.stem)}
                    for f in sorted((s / "examples").glob("*.jpeg"))
                ]
                if (s / "examples").exists()
                else []
            )
            (s / "space_config.json").write_text(
                json.dumps({"model_repo": model_repo, "examples": examples}, indent=2)
            )
            (s / "README.md").write_text(STATIC_README)
            sdk = "static"
        api.create_repo(space_repo, repo_type="space", space_sdk=sdk, exist_ok=True)
        api.upload_folder(
            repo_id=space_repo, repo_type="space", folder_path=s, commit_message="Upload demo"
        )

    print(f"Model : https://huggingface.co/{model_repo}")
    print(f"Space : https://huggingface.co/spaces/{space_repo}")


if __name__ == "__main__":
    main()
