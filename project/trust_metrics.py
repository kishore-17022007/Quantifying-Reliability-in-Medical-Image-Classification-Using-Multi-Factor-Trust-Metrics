import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from models import enable_mc_dropout


# ------------------------------
# Normalization Helpers
# ------------------------------
def min_max_normalize(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Min-max normalize array to [0, 1]."""
    values = np.asarray(values, dtype=np.float32)
    min_val = values.min()
    max_val = values.max()
    denom = max(max_val - min_val, eps)
    return (values - min_val) / denom


def normalize_with_stats(values: np.ndarray, stats: dict[str, float], eps: float = 1e-8) -> np.ndarray:
    """Normalize array to [0, 1] using externally provided min/max stats."""
    values = np.asarray(values, dtype=np.float32)
    min_val = float(stats.get("min", 0.0))
    max_val = float(stats.get("max", 1.0))
    denom = max(max_val - min_val, eps)
    normalized = (values - min_val) / denom
    return np.clip(normalized, 0.0, 1.0)


# ------------------------------
# MC Dropout Uncertainty
# ------------------------------
@torch.no_grad()
def compute_mc_uncertainty(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    passes: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute MC Dropout mean probability and epistemic uncertainty.

    Returns:
        mean_probs: Mean positive-class probability, shape (N,)
        variances: Variance of positive-class probability, shape (N,)
        all_probs: All MC probabilities, shape (passes, N)
    """
    model.eval()
    enable_mc_dropout(model)

    all_probs = []
    for _ in range(passes):
        logits = model(inputs)
        probs = F.softmax(logits, dim=1)[:, 1]
        all_probs.append(probs.detach().cpu().numpy())

    all_probs = np.stack(all_probs, axis=0)
    mean_probs = all_probs.mean(axis=0)
    variances = all_probs.var(axis=0)
    return mean_probs, variances, all_probs


# ------------------------------
# Ensemble Disagreement
# ------------------------------
def compute_ensemble_disagreement(ensemble_probs: np.ndarray) -> np.ndarray:
    """Compute sample-wise disagreement from ensemble probabilities.

    Args:
        ensemble_probs: Array shape (n_models, n_samples)

    Returns:
        disagreement_scores: Variance across models per sample, shape (n_samples,)
    """
    ensemble_probs = np.asarray(ensemble_probs, dtype=np.float32)
    return ensemble_probs.var(axis=0)


# ------------------------------
# Risk-Aware Trust Score
# ------------------------------
def get_risk_weights(class_names: list[str]) -> dict[str, float]:
    """Default clinical risk weights."""
    risk_weights = {name.lower(): 1.0 for name in class_names}
    risk_weights["normal"] = 1.0
    risk_weights["pneumonia"] = 1.5
    return risk_weights


def compute_trust_score(
    uncertainty: np.ndarray,
    ece: float | np.ndarray,
    disagreement: np.ndarray,
    confidence: np.ndarray,
    pred_labels: np.ndarray,
    class_names: list[str],
    weights: dict[str, float] | None = None,
    risk_weights: dict[str, float] | None = None,
    normalization_stats: dict[str, dict[str, float]] | None = None,
) -> np.ndarray:
    """Compute multi-factor trust score.

    Trust = w1*(1-uncertainty) + w2*(1-ece) + w3*(1-disagreement) + w4*(confidence*risk_weight)
    """
    if weights is None:
        weights = {
            "uncertainty": 0.30,
            "ece": 0.20,
            "disagreement": 0.20,
            "confidence": 0.30,
        }

    if risk_weights is None:
        risk_weights = get_risk_weights(class_names)

    if normalization_stats is None:
        uncertainty_n = min_max_normalize(uncertainty)
        disagreement_n = min_max_normalize(disagreement)
        confidence_n = min_max_normalize(confidence)
    else:
        uncertainty_n = normalize_with_stats(uncertainty, normalization_stats["uncertainty"])
        disagreement_n = normalize_with_stats(disagreement, normalization_stats["disagreement"])
        confidence_n = normalize_with_stats(confidence, normalization_stats["confidence"])

    if np.isscalar(ece):
        ece_values = np.full_like(confidence_n, fill_value=float(ece), dtype=np.float32)
    else:
        ece_values = np.asarray(ece, dtype=np.float32)

    if normalization_stats is None:
        ece_n = min_max_normalize(ece_values)
    else:
        ece_n = normalize_with_stats(ece_values, normalization_stats["ece"])

    pred_labels = np.asarray(pred_labels).astype(int)
    class_lookup = {idx: name.lower() for idx, name in enumerate(class_names)}
    sample_risk = np.array([risk_weights.get(class_lookup[label], 1.0) for label in pred_labels], dtype=np.float32)

    trust = (
        weights["uncertainty"] * (1.0 - uncertainty_n)
        + weights["ece"] * (1.0 - ece_n)
        + weights["disagreement"] * (1.0 - disagreement_n)
        + weights["confidence"] * (confidence_n * sample_risk)
    )

    return np.clip(trust, 0.0, 1.5)


def build_normalization_stats(
    uncertainty: np.ndarray,
    ece: float | np.ndarray,
    disagreement: np.ndarray,
    confidence: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Build min/max normalization stats for stable trust scoring across runs."""
    uncertainty = np.asarray(uncertainty, dtype=np.float32)
    disagreement = np.asarray(disagreement, dtype=np.float32)
    confidence = np.asarray(confidence, dtype=np.float32)

    if np.isscalar(ece):
        ece_values = np.array([float(ece)], dtype=np.float32)
    else:
        ece_values = np.asarray(ece, dtype=np.float32)

    return {
        "uncertainty": {"min": float(np.min(uncertainty)), "max": float(np.max(uncertainty))},
        "disagreement": {"min": float(np.min(disagreement)), "max": float(np.max(disagreement))},
        "confidence": {"min": float(np.min(confidence)), "max": float(np.max(confidence))},
        "ece": {"min": float(np.min(ece_values)), "max": float(np.max(ece_values))},
    }


def learn_trust_weights(
    uncertainty: np.ndarray,
    ece: float | np.ndarray,
    disagreement: np.ndarray,
    confidence: np.ndarray,
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
    class_names: list[str],
    risk_weights: dict[str, float],
    normalization_stats: dict[str, dict[str, float]] | None = None,
    step: float = 0.05,
) -> tuple[dict[str, float], float]:
    """Learn trust-factor weights by maximizing AUROC for correctness prediction on validation data."""
    pred_labels = np.asarray(pred_labels).astype(int)
    true_labels = np.asarray(true_labels).astype(int)
    correctness = (pred_labels == true_labels).astype(int)

    if np.unique(correctness).size < 2:
        return {
            "uncertainty": 0.30,
            "ece": 0.20,
            "disagreement": 0.20,
            "confidence": 0.30,
        }, float("nan")

    if normalization_stats is None:
        normalization_stats = build_normalization_stats(
            uncertainty=uncertainty,
            ece=ece,
            disagreement=disagreement,
            confidence=confidence,
        )

    grid = np.arange(0.0, 1.0 + 1e-8, step, dtype=np.float32)

    best_weights = None
    best_auc = -1.0

    for w1 in grid:
        for w2 in grid:
            for w3 in grid:
                w4 = 1.0 - (w1 + w2 + w3)
                if w4 < -1e-8:
                    continue
                if w4 < 0:
                    w4 = 0.0

                weights = {
                    "uncertainty": float(w1),
                    "ece": float(w2),
                    "disagreement": float(w3),
                    "confidence": float(w4),
                }

                trust = compute_trust_score(
                    uncertainty=uncertainty,
                    ece=ece,
                    disagreement=disagreement,
                    confidence=confidence,
                    pred_labels=pred_labels,
                    class_names=class_names,
                    weights=weights,
                    risk_weights=risk_weights,
                    normalization_stats=normalization_stats,
                )

                auc = roc_auc_score(correctness, trust)
                if auc > best_auc:
                    best_auc = float(auc)
                    best_weights = weights

    if best_weights is None:
        best_weights = {
            "uncertainty": 0.30,
            "ece": 0.20,
            "disagreement": 0.20,
            "confidence": 0.30,
        }

    return best_weights, best_auc


def classify_trust_levels(
    trust_scores: np.ndarray,
    pred_labels: np.ndarray,
    class_names: list[str],
    reliable_threshold: float = 0.72,
    review_threshold: float = 0.50,
    high_risk_margin: float = 0.08,
) -> list[str]:
    """Classify trust into Reliable / Review Needed / High Uncertainty.

    For high-risk class (Pneumonia), thresholds are increased by high_risk_margin.
    """
    pred_labels = np.asarray(pred_labels).astype(int)
    trust_scores = np.asarray(trust_scores)
    class_lookup = {idx: name.lower() for idx, name in enumerate(class_names)}

    categories = []
    for score, label in zip(trust_scores, pred_labels):
        class_name = class_lookup[label]
        class_reliable = reliable_threshold
        class_review = review_threshold

        if class_name == "pneumonia":
            class_reliable += high_risk_margin
            class_review += high_risk_margin

        if score >= class_reliable:
            categories.append("Reliable")
        elif score >= class_review:
            categories.append("Review Needed")
        else:
            categories.append("High Uncertainty")

    return categories
