"""Pneumonia-X demo (Gradio): probability of pneumonia on a chest X-ray + Grad-CAM heat map.

Model files (model.safetensors + config.json) are read from MODEL_DIR if set, otherwise
downloaded from the Hugging Face model repo named in model_repo.txt (or env MODEL_REPO).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms as T

from gradcam import GradCAM, overlay

HERE = Path(__file__).resolve().parent


def model_dir() -> Path:
    if os.environ.get("MODEL_DIR"):
        return Path(os.environ["MODEL_DIR"])
    repo = os.environ.get("MODEL_REPO") or (HERE / "model_repo.txt").read_text().strip()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, allow_patterns=["model.safetensors", "config.json"]))


@lru_cache(maxsize=1)
def load(path: str | None = None):
    d = Path(path) if path else model_dir()
    cfg = json.loads((d / "config.json").read_text())
    model = timm.create_model(cfg["timm_name"], pretrained=False, num_classes=1)
    model.load_state_dict(load_file(d / "model.safetensors"))
    model.eval()
    tf = T.Compose(
        [
            T.Resize((cfg["img_size"], cfg["img_size"])),
            T.ToTensor(),
            T.Normalize(cfg["mean"], cfg["std"]),
        ]
    )
    return model, tf, GradCAM(model, model.features), cfg


def predict(image: Image.Image, path: str | None = None):
    if image is None:
        return None, None, ""
    model, tf, cam, cfg = load(path)
    rgb = image.convert("RGB")
    logit, heat = cam(tf(rgb).unsqueeze(0))
    p = float(torch.sigmoid(torch.tensor(logit)))
    t = cfg["threshold"]
    decision = "Pneumonie suspectée" if p >= t else "Pas de pneumonie détectée"
    text = (
        f"**{decision}** — probabilité {p:.1%} (seuil {t:.2f}, choisi pour une sensibilité "
        f"≥ 98 % chez l'enfant).\n\n"
        "La carte montre les zones qui ont fait monter le score (rouge = forte influence). "
        "Elle n'indique pas où se trouve la maladie : un modèle peut s'appuyer sur des indices "
        "sans rapport avec les poumons."
    )
    shown = np.asarray(rgb.resize((512, round(512 * rgb.height / rgb.width))))
    return {"pneumonie": p, "normal": 1 - p}, overlay(shown, heat), text


def build_ui():
    import gradio as gr

    cfg_note = (
        "⚠️ **Démonstration pédagogique, pas un dispositif médical.** "
        "Modèle DenseNet121 entraîné sur des radiographies **d'enfants de 1 à 5 ans** "
        "(jeu de Kermany, un seul hôpital). Testé sur "
        "26 684 radios **d'adultes** (RSNA) : AUROC 0,957, mais avec le seuil pédiatrique, "
        "**22 % des adultes sains et 78 % des radios montrant une autre anomalie déclenchent une "
        "alerte**. Le modèle repère surtout une radio « anormale », "
        "pas spécifiquement une pneumonie."
    )
    examples = sorted(str(p) for p in (HERE / "examples").glob("*.jpeg"))
    with gr.Blocks(title="Pneumonia-X") as ui:
        gr.Markdown("# Pneumonia-X — détection de pneumonie sur radiographie thoracique")
        gr.Markdown(cfg_note)
        with gr.Row():
            inp = gr.Image(type="pil", label="Radiographie thoracique (face)")
            with gr.Column():
                label = gr.Label(label="Probabilité", num_top_classes=2)
                heat = gr.Image(label="Grad-CAM : zones qui ont influencé la décision")
                text = gr.Markdown()
        btn = gr.Button("Analyser", variant="primary")
        btn.click(predict, inputs=inp, outputs=[label, heat, text])
        if examples:
            gr.Examples(
                examples,
                inputs=inp,
                outputs=[label, heat, text],
                fn=predict,
                cache_examples=False,
                run_on_click=True,
            )
        gr.Markdown(
            "Code, données, évaluation interne et externe : "
            "[github.com/khadijafair/pneumonia](https://github.com/khadijafair/pneumonia). "
            "Exemples : jeu de Kermany et al., *Cell* 2018 (CC BY 4.0)."
        )
    return ui


if __name__ == "__main__":
    load()  # fail fast at startup if the model cannot be loaded
    build_ui().launch()
