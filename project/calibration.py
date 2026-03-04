import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ------------------------------
# Calibration Metrics
# ------------------------------
def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error (ECE) for binary classification.

    Args:
        probs: Positive-class probabilities of shape (N,)
        labels: Binary ground-truth labels of shape (N,)
        n_bins: Number of confidence bins
    """
    probs = np.asarray(probs).reshape(-1)
    labels = np.asarray(labels).reshape(-1)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        in_bin = (probs > low) & (probs <= high) if i > 0 else (probs >= low) & (probs <= high)
        if not np.any(in_bin):
            continue

        bin_confidence = probs[in_bin].mean()
        bin_accuracy = labels[in_bin].mean()
        bin_weight = in_bin.mean()
        ece += np.abs(bin_confidence - bin_accuracy) * bin_weight

    return float(ece)


# ------------------------------
# Reliability Diagram
# ------------------------------
def plot_reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
    title: str = "Reliability Diagram",
    save_path: str | None = None,
) -> None:
    """Plot reliability diagram for binary classifier probabilities."""
    probs = np.asarray(probs).reshape(-1)
    labels = np.asarray(labels).reshape(-1)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    accuracies = np.zeros(n_bins)
    confidences = np.zeros(n_bins)

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        in_bin = (probs > low) & (probs <= high) if i > 0 else (probs >= low) & (probs <= high)
        if np.any(in_bin):
            accuracies[i] = labels[in_bin].mean()
            confidences[i] = probs[in_bin].mean()
        else:
            accuracies[i] = np.nan
            confidences[i] = np.nan

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    plt.plot(bin_centers, accuracies, marker="o", linewidth=2, label="Model")
    plt.title(title)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


# ------------------------------
# Temperature Scaling
# ------------------------------
class TemperatureScaler(nn.Module):
    """Single-parameter temperature scaling module for logits calibration."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = torch.clamp(self.temperature, min=1e-3)
        return logits / temperature


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 200,
    lr: float = 0.01,
    device: str | torch.device = "cpu",
) -> float:
    """Fit temperature parameter on validation logits using NLL minimization."""
    scaler = TemperatureScaler().to(device)
    logits = logits.to(device)
    labels = labels.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS(scaler.parameters(), lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = criterion(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.clamp(scaler.temperature.detach(), min=1e-3).item())


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply learned temperature to logits."""
    temperature = max(temperature, 1e-3)
    return logits / temperature
