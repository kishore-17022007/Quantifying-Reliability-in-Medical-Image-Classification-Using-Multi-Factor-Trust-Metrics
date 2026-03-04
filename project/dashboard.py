import json
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import matplotlib.cm as cm
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from calibration import apply_temperature
from models import MCDropoutResNet18, enable_mc_dropout, load_checkpoint
from trust_metrics import classify_trust_levels, compute_trust_score
from utils import GradCAM, get_device, overlay_heatmap_on_image, preprocess_pil_image


# ------------------------------
# Streamlit App Setup
# ------------------------------
st.set_page_config(page_title="Medical Image Trust Dashboard", layout="wide")
st.title("Quantifying Reliability in Medical Image Classification")
st.caption("MC Dropout + Calibration + Ensemble Disagreement + Clinical Risk Weighting")


# ------------------------------
# Sidebar Configuration
# ------------------------------
st.sidebar.header("Configuration")
artifacts_dir = st.sidebar.text_input("Artifacts Directory", value="./artifacts")
base_model_path = st.sidebar.text_input("Base Model Path", value=f"{artifacts_dir}/best_model.pt")
summary_path = st.sidebar.text_input("Summary JSON Path", value=f"{artifacts_dir}/research_summary.json")

default_ensemble = [
    f"{artifacts_dir}/ensemble_model_seed_42.pt",
    f"{artifacts_dir}/ensemble_model_seed_123.pt",
    f"{artifacts_dir}/ensemble_model_seed_999.pt",
]
ensemble_paths = []
for idx, p in enumerate(default_ensemble, start=1):
    ensemble_paths.append(st.sidebar.text_input(f"Ensemble Model {idx}", value=p))

mc_passes = st.sidebar.slider("MC Dropout Passes", min_value=10, max_value=50, value=30, step=5)
gradcam_alpha = st.sidebar.slider("Grad-CAM Overlay Alpha", 0.10, 0.90, 0.45, 0.05)
gradcam_contrast = st.sidebar.slider("Grad-CAM Contrast", 0.50, 3.00, 1.60, 0.10)

w1 = st.sidebar.slider("w1: Uncertainty", 0.0, 1.0, 0.30, 0.05)
w2 = st.sidebar.slider("w2: ECE", 0.0, 1.0, 0.20, 0.05)
w3 = st.sidebar.slider("w3: Disagreement", 0.0, 1.0, 0.20, 0.05)
w4 = st.sidebar.slider("w4: Confidence", 0.0, 1.0, 0.30, 0.05)


# ------------------------------
# Loading Helpers
# ------------------------------
@st.cache_resource
def load_models(base_model_path: str, ensemble_paths: list[str], class_names: list[str], dropout: float):
    device = get_device()

    base_model = MCDropoutResNet18(num_classes=len(class_names), dropout_p=dropout, pretrained=False).to(device)
    load_checkpoint(base_model_path, base_model, map_location=device)
    base_model.eval()

    ensemble_models = []
    for path in ensemble_paths:
        m = MCDropoutResNet18(num_classes=len(class_names), dropout_p=dropout, pretrained=False).to(device)
        load_checkpoint(path, m, map_location=device)
        m.eval()
        ensemble_models.append(m)

    return base_model, ensemble_models, device


def load_summary(summary_path: str) -> dict:
    if not os.path.exists(summary_path):
        return {
            "class_names": ["NORMAL", "PNEUMONIA"],
            "temperature": 1.0,
            "ece_after": 0.0,
            "dropout": 0.3,
        }
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "class_names" not in data:
        data["class_names"] = ["NORMAL", "PNEUMONIA"]
    if "temperature" not in data:
        data["temperature"] = 1.0
    if "ece_after" not in data:
        data["ece_after"] = 0.0
    if "dropout" not in data:
        data["dropout"] = 0.3
    if "decision_threshold" not in data:
        data["decision_threshold"] = 0.5

    return data


# ------------------------------
# Inference Utilities
# ------------------------------
@torch.no_grad()
def mc_dropout_single(model, x, passes=30):
    model.eval()
    probs = []

    for _ in range(passes):
        enable_mc_dropout(model)
        logits = model(x)
        prob = F.softmax(logits, dim=1)[:, 1].item()
        probs.append(prob)

    probs = np.array(probs, dtype=np.float32)
    return float(probs.mean()), float(probs.var())


@torch.no_grad()
def ensemble_single(ensemble_models, x):
    probs = []
    for model in ensemble_models:
        logits = model(x)
        prob = F.softmax(logits, dim=1)[:, 1].item()
        probs.append(prob)
    probs = np.array(probs, dtype=np.float32)
    return float(probs.mean()), float(probs.var())


def trust_gauge(value: float):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": "Final Trust Score"},
            gauge={
                "axis": {"range": [0, 1.2]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 0.5], "color": "#ffcccc"},
                    {"range": [0.5, 0.75], "color": "#fff3cd"},
                    {"range": [0.75, 1.2], "color": "#d4edda"},
                ],
            },
        )
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=60, b=10))
    return fig


# ------------------------------
# Main App Flow
# ------------------------------
summary = load_summary(summary_path)
class_names = summary.get("class_names", ["NORMAL", "PNEUMONIA"])
temperature = float(summary.get("temperature", 1.0))
ece_value = float(summary.get("ece_after", 0.0))
dropout = float(summary.get("dropout", 0.3))
decision_threshold = float(summary.get("decision_threshold", 0.5))

required_files = [base_model_path, *ensemble_paths]
if not all(Path(p).exists() for p in required_files):
    st.warning("Model files are missing. Train first with train.py and update sidebar paths.")

uploaded = st.file_uploader("Upload Chest X-ray image", type=["png", "jpg", "jpeg"])

if uploaded is not None and all(Path(p).exists() for p in required_files):
    image = Image.open(uploaded).convert("RGB")

    base_model, ensemble_models, device = load_models(base_model_path, ensemble_paths, class_names, dropout)

    x = preprocess_pil_image(image).to(device)

    # Base logits + temperature-scaled confidence
    with torch.no_grad():
        logits = base_model(x)
        scaled_logits = apply_temperature(logits, temperature)
        scaled_probs = F.softmax(scaled_logits, dim=1).detach().cpu().numpy()[0]

    pred_idx = 1 if float(scaled_probs[1]) >= decision_threshold else 0
    pred_class = class_names[pred_idx]
    confidence = float(scaled_probs[pred_idx])

    # MC Dropout uncertainty
    mc_mean, mc_uncertainty = mc_dropout_single(base_model, x, passes=mc_passes)

    # Ensemble disagreement
    ensemble_mean, ensemble_disagreement = ensemble_single(ensemble_models, x)

    # Trust score
    trust_score = compute_trust_score(
        uncertainty=np.array([mc_uncertainty]),
        ece=ece_value,
        disagreement=np.array([ensemble_disagreement]),
        confidence=np.array([confidence]),
        pred_labels=np.array([pred_idx]),
        class_names=class_names,
        weights={"uncertainty": w1, "ece": w2, "disagreement": w3, "confidence": w4},
        risk_weights={"normal": 1.0, "pneumonia": 1.5},
    )[0]

    trust_level = classify_trust_levels(
        trust_scores=np.array([trust_score]),
        pred_labels=np.array([pred_idx]),
        class_names=class_names,
    )[0]

    # Grad-CAM
    cam_extractor = GradCAM(base_model, base_model.backbone.layer3[-1].conv2)
    cam = cam_extractor(x, target_class=pred_idx)
    cam_extractor.release()

    # Improve visibility of salient regions with contrast adjustment
    cam_visible = np.power(np.clip(cam, 0.0, 1.0), 1.0 / max(gradcam_contrast, 1e-6))
    cam_color = (cm.jet(cam_visible)[..., :3] * 255).astype(np.uint8)
    cam_color = np.asarray(
        Image.fromarray(cam_color).resize((224, 224), resample=Image.BILINEAR)
    )
    overlay = overlay_heatmap_on_image(image.resize((224, 224)), cam_visible, alpha=gradcam_alpha)

    # Dashboard layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input and Grad-CAM")
        h1, h2, h3 = st.columns(3)
        h1.image(image.resize((224, 224)), caption="Uploaded Image", use_container_width=True)
        h2.image(cam_color, caption="Raw Grad-CAM Heatmap", use_container_width=True)
        h3.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)

    with col2:
        st.subheader("Prediction and Reliability")
        st.markdown(f"**Prediction:** {pred_class}")
        st.markdown(f"**Decision Threshold (Pneumonia):** {decision_threshold:.2f}")
        st.markdown(f"**Confidence:** {confidence:.4f}")
        st.markdown(f"**MC Uncertainty (Variance):** {mc_uncertainty:.6f}")
        st.markdown(f"**ECE (Calibrated):** {ece_value:.6f}")
        st.markdown(f"**Ensemble Disagreement:** {ensemble_disagreement:.6f}")
        st.markdown(f"**Trust Category:** {trust_level}")

        st.plotly_chart(trust_gauge(float(trust_score)), use_container_width=True)

    st.divider()
    st.subheader("Auxiliary Estimates")
    c1, c2 = st.columns(2)
    c1.metric("MC Mean Probability (Pneumonia)", f"{mc_mean:.4f}")
    c2.metric("Ensemble Mean Probability (Pneumonia)", f"{ensemble_mean:.4f}")

elif uploaded is None:
    st.info("Upload a chest X-ray image to run trust-aware inference.")
