"""
Model definitions for crop disease classification.
Supports EfficientNet-B0/B3, ResNet50/34, MobileNetV3-Small/Large.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Dict


class CropDiseaseClassifier(nn.Module):
    """
    Transfer-learning classifier with a configurable backbone.
    The backbone's final classification layer is replaced with a custom head.
    """

    SUPPORTED = {
        "efficientnet_b0",
        "efficientnet_b3",
        "resnet50",
        "resnet34",
        "mobilenet_v3_small",
        "mobilenet_v3_large",
    }

    def __init__(
        self,
        architecture: str = "mobilenet_v3_small",
        num_classes: int = 38,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.architecture = architecture
        self.num_classes = num_classes

        weights_flag = "DEFAULT" if pretrained else None
        backbone, in_features = self._build_backbone(architecture, weights_flag)
        self.backbone = backbone

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

    def _build_backbone(self, arch: str, weights) -> tuple:
        if arch == "efficientnet_b0":
            backbone = models.efficientnet_b0(weights=weights)
            in_features = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
        elif arch == "efficientnet_b3":
            backbone = models.efficientnet_b3(weights=weights)
            in_features = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
        elif arch == "resnet50":
            backbone = models.resnet50(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif arch == "resnet34":
            backbone = models.resnet34(weights=weights)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif arch == "mobilenet_v3_small":
            backbone = models.mobilenet_v3_small(weights=weights)
            # classifier[0] is the first Linear (576 → 1024); that's what the backbone outputs
            in_features = backbone.classifier[0].in_features
            backbone.classifier = nn.Identity()
        elif arch == "mobilenet_v3_large":
            backbone = models.mobilenet_v3_large(weights=weights)
            # classifier[0] is the first Linear (960 → 1280)
            in_features = backbone.classifier[0].in_features
            backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported architecture: {arch}. Choose from {self.SUPPORTED}")

        return backbone, in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def freeze_backbone(self):
        """Freeze backbone weights (train only classifier head)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all weights for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_num_params(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


def build_model(
    architecture: str = "mobilenet_v3_small",
    num_classes: int = 38,
    pretrained: bool = True,
    dropout: float = 0.3,
    device: str = "auto",
) -> tuple:
    """
    Build and move model to device.
    Returns (model, device_str).
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CropDiseaseClassifier(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )
    model = model.to(device)

    params = model.get_num_params()
    print(f"Model: {architecture} | Params: {params['total']:,} total, {params['trainable']:,} trainable")
    print(f"Device: {device.upper()}")

    return model, device


def load_model(checkpoint_path: str, device: str = "auto") -> tuple:
    """
    Load model from checkpoint.
    Returns (model, class_to_idx, metadata).
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = CropDiseaseClassifier(
        architecture=checkpoint["architecture"],
        num_classes=checkpoint["num_classes"],
        pretrained=False,
        dropout=checkpoint.get("dropout", 0.3),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, checkpoint["class_to_idx"], checkpoint.get("metadata", {})
