"""
Evaluation utilities: metrics, confusion matrix, Grad-CAM visualizations.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)
from tqdm import tqdm


# ─── Core metrics ──────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_metrics(
    model: nn.Module,
    loader,
    device: str,
    class_names: List[str],
) -> Dict:
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(all_labels, all_preds)

    return {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "f1_macro": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(all_labels, all_preds, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(all_labels, all_preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(all_labels, all_preds, average="macro", zero_division=0)),
        "per_class": report,
        "confusion_matrix": cm.tolist(),
    }


# ─── Grad-CAM ──────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Highlights the image regions most influential for a given prediction.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._gradients = None
        self._activations = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, input, output):
            self._activations = output.detach()

        def bwd_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(fwd_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(bwd_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.eval()
        input_tensor = input_tensor.unsqueeze(0) if input_tensor.dim() == 3 else input_tensor
        input_tensor.requires_grad_(True)

        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Global average pooling of gradients
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam


def get_gradcam_layer(model) -> Optional[nn.Module]:
    """Auto-select the last conv/feature block for Grad-CAM."""
    arch = getattr(model, "architecture", "")
    try:
        if "efficientnet" in arch:
            return model.backbone.features[-1]
        elif "resnet" in arch:
            return model.backbone.layer4[-1]
        elif "mobilenet" in arch:
            return model.backbone.features[-1]
    except (AttributeError, IndexError):
        pass
    return None


# ─── Prediction with confidence ───────────────────────────────────────────────

@torch.no_grad()
def predict_single(
    model: nn.Module,
    image_tensor: torch.Tensor,
    idx_to_class: Dict[int, str],
    device: str,
    top_k: int = 5,
) -> List[Dict]:
    """
    Run inference on a single image tensor.
    Returns list of top-k predictions with class name, index, and probability.
    """
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)
    logits = model(image_tensor)
    probs = torch.softmax(logits, dim=1)[0]

    top_probs, top_indices = probs.topk(min(top_k, len(idx_to_class)))
    return [
        {
            "class_name": idx_to_class[idx.item()],
            "class_index": idx.item(),
            "probability": float(prob.item()),
        }
        for prob, idx in zip(top_probs, top_indices)
    ]


# ─── Visualizations (saved to disk) ───────────────────────────────────────────

def plot_training_history(history_path: str, output_dir: str):
    """Generate training curves from saved history JSON."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with open(history_path) as f:
            history = json.load(f)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss plot
        axes[0].plot(history["train_loss"], label="Train Loss", color="steelblue")
        axes[0].plot(history["val_loss"], label="Val Loss", color="coral")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Accuracy plot
        axes[1].plot(history["train_acc"], label="Train Acc", color="steelblue")
        axes[1].plot(history["val_acc"], label="Val Acc", color="coral")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Training curves saved to {output_dir}/training_curves.png")
    except ImportError:
        print("matplotlib not available; skipping training curve plot.")


def plot_confusion_matrix(
    cm: List[List[int]],
    class_names: List[str],
    output_dir: str,
    normalize: bool = True,
):
    """Save confusion matrix heatmap."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = np.array(cm)
        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            cm = cm / np.where(row_sums == 0, 1, row_sums)

        fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 0.5),
                                        max(8, len(class_names) * 0.4)))
        sns.heatmap(
            cm, annot=len(class_names) <= 20, fmt=".2f" if normalize else "d",
            cmap="Blues", xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""))
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.yticks(rotation=0, fontsize=7)
        plt.tight_layout()

        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Confusion matrix saved to {output_dir}/confusion_matrix.png")
    except ImportError:
        print("matplotlib/seaborn not available; skipping confusion matrix plot.")
