import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def collect_ensemble_probabilities(
    models: list[torch.nn.Module],
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect positive-class probabilities from an ensemble on a dataloader.

    Returns:
        ensemble_probs: shape (n_models, n_samples)
        labels: shape (n_samples,)
    """
    labels_all = []
    model_probs = []

    for model in models:
        model.eval()
        probs_this_model = []

        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)[:, 1]
            probs_this_model.append(probs.detach().cpu().numpy())

            if len(model_probs) == 0:
                labels_all.append(labels.numpy())

        model_probs.append(np.concatenate(probs_this_model, axis=0))

    labels_all = np.concatenate(labels_all, axis=0)
    ensemble_probs = np.stack(model_probs, axis=0)
    return ensemble_probs, labels_all


def disagreement_from_ensemble_probs(ensemble_probs: np.ndarray) -> np.ndarray:
    """Variance-based ensemble disagreement score per sample."""
    ensemble_probs = np.asarray(ensemble_probs, dtype=np.float32)
    return ensemble_probs.var(axis=0)
