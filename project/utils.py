import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ------------------------------
# Reproducibility and Device
# ------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------
# Data Pipeline
# ------------------------------
def build_transforms(image_size: int = 224):
    train_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.08, contrast=0.08),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    eval_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return train_tfms, eval_tfms


def _check_standard_structure(data_root: Path) -> bool:
    return all((data_root / split).exists() for split in ["train", "val", "test"])


def _make_val_split(train_dir: Path, train_tfms, eval_tfms, val_split: float, seed: int):
    full_train = datasets.ImageFolder(str(train_dir), transform=train_tfms)
    full_train_eval = datasets.ImageFolder(str(train_dir), transform=eval_tfms)

    indices = np.arange(len(full_train))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    split_idx = int(len(indices) * (1 - val_split))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_set = Subset(full_train, train_indices.tolist())
    val_set = Subset(full_train_eval, val_indices.tolist())
    return train_set, val_set, full_train.classes


def get_dataloaders(
    data_root: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 2,
    val_split: float = 0.15,
    seed: int = 42,
):
    """Build train/val/test dataloaders for Chest X-ray Pneumonia dataset.

    Supports:
    1) Standard Kaggle structure with train/val/test folders.
    2) train/test only, where val is split from train.
    """
    data_root = Path(data_root)
    train_tfms, eval_tfms = build_transforms(image_size=image_size)

    if _check_standard_structure(data_root):
        train_set = datasets.ImageFolder(str(data_root / "train"), transform=train_tfms)
        val_set = datasets.ImageFolder(str(data_root / "val"), transform=eval_tfms)
        test_set = datasets.ImageFolder(str(data_root / "test"), transform=eval_tfms)
        class_names = train_set.classes
    else:
        train_dir = data_root / "train"
        test_dir = data_root / "test"
        if not train_dir.exists() or not test_dir.exists():
            raise FileNotFoundError(
                "Dataset not found. Expected either train/val/test or train/test under data_root."
            )
        train_set, val_set, class_names = _make_val_split(
            train_dir, train_tfms, eval_tfms, val_split=val_split, seed=seed
        )
        test_set = datasets.ImageFolder(str(test_dir), transform=eval_tfms)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_names


# ------------------------------
# Metrics and Logging
# ------------------------------
def binary_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    brier = float(np.mean((y_prob - y_true) ** 2))

    return {
        "accuracy": float(accuracy),
        "f1": float(f1),
        "auc": float(auc),
        "brier": float(brier),
    }


def optimize_binary_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Find threshold in [0.05, 0.95] maximizing F1 score on validation data."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    thresholds = np.linspace(0.05, 0.95, 91)
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)

    return best_threshold, best_f1


def save_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ------------------------------
# Plotting Helpers
# ------------------------------
def plot_distribution(values: np.ndarray, title: str, xlabel: str, save_path: str) -> None:
    plt.figure(figsize=(7, 5))
    plt.hist(values, bins=30, alpha=0.85)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, save_path: str) -> None:
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ------------------------------
# Inference and Grad-CAM utilities
# ------------------------------
def preprocess_pil_image(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    """Preprocess PIL image for ResNet18 input."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    tfm = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return tfm(image).unsqueeze(0)


class GradCAM:
    """Grad-CAM implementation for MCDropoutResNet18 backbone layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        score = logits[:, target_class].sum()
        score.backward(retain_graph=True)

        grads = self.gradients
        acts = self.activations
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam[0]

        cam -= cam.min()
        cam /= cam.max() + 1e-8
        return cam.detach().cpu().numpy()

    def release(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def overlay_heatmap_on_image(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """Overlay normalized heatmap onto RGB image and return numpy array."""
    image = image.convert("RGB")
    image_np = np.asarray(image).astype(np.float32) / 255.0

    # Upsample low-resolution CAM to image resolution for sharper display
    heatmap_img = Image.fromarray((np.clip(heatmap, 0.0, 1.0) * 255).astype(np.uint8))
    heatmap_img = heatmap_img.resize((image_np.shape[1], image_np.shape[0]), resample=Image.BILINEAR)
    heatmap = np.asarray(heatmap_img).astype(np.float32) / 255.0

    cmap = plt.get_cmap("jet")
    heatmap_rgb = cmap(heatmap)[..., :3]

    blended = (1 - alpha) * image_np + alpha * heatmap_rgb
    blended = np.clip(blended, 0.0, 1.0)
    return (blended * 255).astype(np.uint8)
