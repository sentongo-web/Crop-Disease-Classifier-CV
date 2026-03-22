"""
Training script for crop disease classifier.

Usage:
    python src/train.py --data ./data/plantvillage --epochs 10
    python src/train.py --data ./data/sample --arch mobilenet_v3_small --epochs 5 --batch-size 8
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, ReduceLROnPlateau, StepLR, OneCycleLR
)
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import build_model
from src.dataset import get_dataloaders
from src.evaluate import compute_metrics


# ─── Training utilities ────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: float = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class MetricsTracker:
    def __init__(self):
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    def update(self, train_loss, train_acc, val_loss, val_acc, lr):
        self.history["train_loss"].append(train_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)
        self.history["lr"].append(lr)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)


# ─── Train / eval loops ────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer,
    device: str,
) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.3f}")

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


# ─── Checkpoint utilities ──────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    path: str,
    epoch: int,
    val_acc: float,
    class_to_idx: Dict,
    args,
    metadata: Dict = None,
):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "architecture": args.arch,
        "num_classes": len(class_to_idx),
        "dropout": args.dropout,
        "class_to_idx": class_to_idx,
        "epoch": epoch,
        "val_acc": val_acc,
        "metadata": metadata or {},
    }, path)


# ─── Main training function ────────────────────────────────────────────────────

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(" Loading dataset...")
    print("="*60)
    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(
        data_dir=args.data,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_split,
        val_ratio=args.val_split,
        use_weighted_sampler=True,
        seed=args.seed,
    )
    num_classes = len(class_to_idx)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(" Building model...")
    print("="*60)
    model, device = build_model(
        architecture=args.arch,
        num_classes=num_classes,
        pretrained=args.pretrained,
        dropout=args.dropout,
    )

    # Phase 1: freeze backbone, train only head
    if args.freeze_epochs > 0:
        print(f"\nPhase 1: Training head only for {args.freeze_epochs} epochs (backbone frozen)")
        model.freeze_backbone()

    # ── Loss & Optimizer ──────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ── Scheduler ─────────────────────────────────────────────────────────────
    total_steps = args.epochs * len(train_loader)
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.scheduler == "onecycle":
        scheduler = OneCycleLR(optimizer, max_lr=args.lr * 10, total_steps=total_steps)
    elif args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    else:
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    # ── Training loop ─────────────────────────────────────────────────────────
    early_stopping = EarlyStopping(patience=args.patience)
    tracker = MetricsTracker()
    best_val_acc = 0.0
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print(f" Training for {args.epochs} epochs on {device.upper()}")
    print("="*60)

    for epoch in range(1, args.epochs + 1):
        # Phase 2 switch: unfreeze backbone
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs + 1:
            print(f"\nPhase 2: Unfreezing backbone for fine-tuning (epoch {epoch})")
            model.unfreeze_backbone()
            # Rebuild optimizer with lower LR for backbone
            optimizer = AdamW([
                {"params": model.backbone.parameters(), "lr": args.lr * 0.1},
                {"params": model.classifier.parameters(), "lr": args.lr},
            ], weight_decay=args.weight_decay)
            if args.scheduler == "cosine":
                scheduler = CosineAnnealingLR(
                    optimizer, T_max=args.epochs - args.freeze_epochs, eta_min=1e-6
                )

        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        tracker.update(train_loss, train_acc, val_loss, val_acc, current_lr)

        if args.scheduler in ("cosine", "step"):
            scheduler.step()
        elif args.scheduler == "plateau":
            scheduler.step(val_loss)

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch:3d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"lr={current_lr:.2e} ({elapsed:.1f}s)"
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, str(save_dir / "best_model.pth"),
                epoch, val_acc, class_to_idx, args,
                metadata={"best_epoch": epoch, "best_val_acc": float(val_acc)},
            )
            print(f"  [SAVED] Best model (val_acc={val_acc:.4f})")

        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break

    # ── Save final & history ──────────────────────────────────────────────────
    save_checkpoint(
        model, str(save_dir / "last_model.pth"),
        epoch, val_acc, class_to_idx, args,
    )
    tracker.save(str(save_dir / "training_history.json"))

    print(f"\n{'='*60}")
    print(f" Training complete! Best val_acc: {best_val_acc:.4f}")
    print(f" Models saved to: {save_dir}/")
    print("="*60)

    # ── Test set evaluation ───────────────────────────────────────────────────
    print("\nEvaluating best model on test set...")
    checkpoint = torch.load(str(save_dir / "best_model.pth"), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    metrics = compute_metrics(model, test_loader, device, list(class_to_idx.keys()))
    print(f"\nTest Results:")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted): {metrics['f1_weighted']:.4f}")

    with open(str(save_dir / "test_metrics.json"), "w") as f:
        # confusion_matrix contains numpy arrays; convert to lists
        safe_metrics = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
        safe_metrics["confusion_matrix"] = metrics["confusion_matrix"]
        json.dump(safe_metrics, f, indent=2)

    return best_val_acc


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    import sys as _sys
    parser = argparse.ArgumentParser(description="Train Crop Disease Classifier")
    parser.add_argument("--data", required=True, help="Path to dataset directory")
    parser.add_argument("--arch", default="mobilenet_v3_small",
                        choices=["efficientnet_b0", "efficientnet_b3", "resnet50",
                                 "resnet34", "mobilenet_v3_small", "mobilenet_v3_large"],
                        help="Model architecture")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--scheduler", default="cosine",
                        choices=["cosine", "plateau", "step", "onecycle"])
    parser.add_argument("--freeze-epochs", type=int, default=3,
                        help="Epochs to train only the head before unfreezing backbone")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (0 = main process, recommended on Windows)")
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--save-dir", default="./models")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
