"""
Dataset classes and data loading utilities for crop disease classification.
"""

import os
import random
import sys
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# On Windows, multiprocessing workers cause issues; default to 0
_DEFAULT_WORKERS = 0 if sys.platform == "win32" else 4


# ─── Albumentations transforms ────────────────────────────────────────────────

def get_train_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.OneOf([
            A.MotionBlur(p=0.5),
            A.MedianBlur(blur_limit=3, p=0.5),
            A.GaussianBlur(p=0.5),
        ], p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.CLAHE(p=0.3),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 32),
            hole_width_range=(8, 32),
            p=0.3,
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(height=image_size, width=image_size, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_inference_transforms(image_size: int = 224) -> A.Compose:
    return get_val_transforms(image_size)


# ─── Dataset class ─────────────────────────────────────────────────────────────

class CropDiseaseDataset(Dataset):
    """
    Dataset for crop disease classification.
    Expects directory structure:
        root/
            class_name_1/
                img1.jpg
                img2.jpg
            class_name_2/
                ...
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

    def __init__(
        self,
        root: str,
        transform: Optional[A.Compose] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ):
        self.root = Path(root)
        self.transform = transform

        # Build class list (sorted for reproducibility)
        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        else:
            classes = sorted([
                d.name for d in self.root.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ])
            self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.classes = list(self.class_to_idx.keys())

        self.samples: List[Tuple[str, int]] = []
        for cls_name, cls_idx in self.class_to_idx.items():
            cls_dir = self.root / cls_name
            if not cls_dir.exists():
                continue
            for img_path in cls_dir.iterdir():
                if img_path.suffix in self.EXTENSIONS:
                    self.samples.append((str(img_path), cls_idx))

        if len(self.samples) == 0:
            raise ValueError(f"No images found in {root}. Check directory structure.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for balanced sampling."""
        counts = [0] * len(self.classes)
        for _, label in self.samples:
            counts[label] += 1
        counts = np.array(counts, dtype=np.float32)
        counts = np.where(counts == 0, 1, counts)
        weights = 1.0 / counts
        weights = weights / weights.sum()
        return torch.tensor(weights, dtype=torch.float32)

    def get_sample_weights(self) -> torch.Tensor:
        """Per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights()
        return torch.tensor([class_weights[label] for _, label in self.samples])


# ─── Split utilities ───────────────────────────────────────────────────────────

def split_dataset(
    root: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List, List, List, Dict]:
    """
    Split image paths per class into train/val/test.
    Returns (train_samples, val_samples, test_samples, class_to_idx).
    """
    random.seed(seed)
    root_path = Path(root)
    classes = sorted([d.name for d in root_path.iterdir() if d.is_dir() and not d.name.startswith(".")])
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    train, val, test = [], [], []

    for cls_name, cls_idx in class_to_idx.items():
        cls_dir = root_path / cls_name
        if not cls_dir.exists():
            continue
        paths = [str(p) for p in cls_dir.iterdir() if p.suffix in EXTENSIONS]
        random.shuffle(paths)

        n = len(paths)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train += [(p, cls_idx) for p in paths[:n_train]]
        val += [(p, cls_idx) for p in paths[n_train:n_train + n_val]]
        test += [(p, cls_idx) for p in paths[n_train + n_val:]]

    return train, val, test, class_to_idx


class SampledDataset(Dataset):
    """Dataset from pre-split list of (path, label) tuples."""

    def __init__(self, samples: List[Tuple[str, int]], transform: Optional[A.Compose] = None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label


# ─── DataLoader factory ────────────────────────────────────────────────────────

def get_dataloaders(
    data_dir: str,
    image_size: int = 224,
    batch_size: int = 16,
    num_workers: int = _DEFAULT_WORKERS,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    use_weighted_sampler: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Returns (train_loader, val_loader, test_loader, class_to_idx).
    """
    train_samples, val_samples, test_samples, class_to_idx = split_dataset(
        data_dir, train_ratio, val_ratio, seed
    )

    train_dataset = SampledDataset(train_samples, get_train_transforms(image_size))
    val_dataset = SampledDataset(val_samples, get_val_transforms(image_size))
    test_dataset = SampledDataset(test_samples, get_val_transforms(image_size))

    # pin_memory only helps on GPU; disable on CPU to avoid overhead
    use_pin_memory = torch.cuda.is_available()

    if use_weighted_sampler:
        labels = [s[1] for s in train_samples]
        counts = np.bincount(labels, minlength=len(class_to_idx)).astype(np.float32)
        counts = np.where(counts == 0, 1, counts)
        class_w = 1.0 / counts
        sample_weights = torch.tensor([class_w[l] for l in labels])
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=use_pin_memory,
            persistent_workers=(num_workers > 0),
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=use_pin_memory,
            persistent_workers=(num_workers > 0),
        )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin_memory,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin_memory,
        persistent_workers=(num_workers > 0),
    )

    print(f"Dataset splits — Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"Classes ({len(class_to_idx)}): {list(class_to_idx.keys())[:5]} ...")

    return train_loader, val_loader, test_loader, class_to_idx
