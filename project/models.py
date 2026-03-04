import torch
import torch.nn as nn
import warnings
from torchvision.models import ResNet18_Weights, resnet18


class MCDropoutResNet18(nn.Module):
    """ResNet18 backbone with dropout-enabled classifier head for MC Dropout."""

    def __init__(self, num_classes: int = 2, dropout_p: float = 0.3, pretrained: bool = True):
        super().__init__()

        # Backbone
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        if weights is not None:
            try:
                backbone = resnet18(weights=weights)
            except Exception as exc:
                warnings.warn(
                    f"Could not load pretrained ResNet18 weights ({exc}). Falling back to untrained weights.",
                    RuntimeWarning,
                )
                backbone = resnet18(weights=None)
        else:
            backbone = resnet18(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = self.dropout(features)
        logits = self.classifier(features)
        return logits


def enable_mc_dropout(model: nn.Module) -> None:
    """Enable dropout layers during inference for stochastic forward passes."""
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


def save_checkpoint(path: str, model: nn.Module, extra: dict | None = None) -> None:
    """Save model state dict and metadata."""
    payload = {"model_state_dict": model.state_dict()}
    if extra is not None:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str, model: nn.Module, map_location: str | torch.device = "cpu") -> dict:
    """Load checkpoint into model and return full checkpoint dictionary."""
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint
