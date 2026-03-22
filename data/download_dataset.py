"""
Dataset downloader for PlantVillage dataset.
Supports downloading via Kaggle API or HuggingFace datasets.
"""

import os
import sys
import zipfile
import shutil
import argparse
from pathlib import Path


def download_via_kaggle(output_dir: str):
    """Download PlantVillage dataset from Kaggle."""
    try:
        import kaggle
        print("Downloading PlantVillage dataset from Kaggle...")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "emmarex/plantdisease",
            path=output_dir,
            unzip=True,
            quiet=False,
        )
        print(f"Dataset downloaded to {output_dir}")
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        print("Make sure you have kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME and KAGGLE_KEY env vars.")
        sys.exit(1)


def download_via_huggingface(output_dir: str):
    """Download PlantVillage dataset from HuggingFace."""
    try:
        from datasets import load_dataset
        print("Downloading PlantVillage dataset from HuggingFace...")
        dataset = load_dataset("sasha/plantvillage-data", split="train")

        os.makedirs(output_dir, exist_ok=True)
        label_names = dataset.features["label"].names

        for label_name in label_names:
            os.makedirs(os.path.join(output_dir, label_name), exist_ok=True)

        print(f"Saving {len(dataset)} images...")
        for idx, item in enumerate(dataset):
            label_name = label_names[item["label"]]
            img_path = os.path.join(output_dir, label_name, f"{idx}.jpg")
            item["image"].save(img_path)
            if idx % 1000 == 0:
                print(f"  Saved {idx}/{len(dataset)} images...")

        print(f"Dataset saved to {output_dir}")
    except Exception as e:
        print(f"HuggingFace download failed: {e}")
        print("Try: pip install datasets")
        sys.exit(1)


def verify_dataset(data_dir: str):
    """Verify dataset structure and print statistics."""
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Dataset directory not found: {data_dir}")
        return False

    classes = [d for d in data_path.iterdir() if d.is_dir()]
    print(f"\nDataset Statistics:")
    print(f"  Total classes: {len(classes)}")

    total_images = 0
    for cls in sorted(classes):
        images = list(cls.glob("*.jpg")) + list(cls.glob("*.JPG")) + \
                 list(cls.glob("*.png")) + list(cls.glob("*.PNG"))
        total_images += len(images)
        print(f"  {cls.name}: {len(images)} images")

    print(f"\n  Total images: {total_images}")
    return True


def create_sample_dataset(output_dir: str, num_classes: int = 5, images_per_class: int = 50):
    """Create a small synthetic dataset for testing (no download required)."""
    import numpy as np
    from PIL import Image
    import random

    print(f"Creating sample dataset with {num_classes} classes, {images_per_class} images each...")

    sample_classes = [
        "Tomato___healthy",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Potato___Early_blight",
        "Corn_(maize)___Common_rust_",
    ][:num_classes]

    os.makedirs(output_dir, exist_ok=True)

    for class_name in sample_classes:
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        for i in range(images_per_class):
            # Create synthetic leaf-like images with different color profiles
            img_array = np.zeros((224, 224, 3), dtype=np.uint8)

            # Base green color for leaf
            base_green = [random.randint(30, 80), random.randint(100, 180), random.randint(30, 80)]

            # Add disease patterns based on class
            if "blight" in class_name.lower():
                # Brown spots for blight
                img_array[:, :] = base_green
                for _ in range(random.randint(5, 20)):
                    x, y = random.randint(0, 200), random.randint(0, 200)
                    r = random.randint(5, 25)
                    for dx in range(-r, r):
                        for dy in range(-r, r):
                            if dx*dx + dy*dy <= r*r and 0 <= x+dx < 224 and 0 <= y+dy < 224:
                                img_array[x+dx, y+dy] = [139, 90, 43]
            elif "rust" in class_name.lower():
                # Orange spots for rust
                img_array[:, :] = base_green
                for _ in range(random.randint(10, 30)):
                    x, y = random.randint(0, 220), random.randint(0, 220)
                    img_array[x:x+4, y:y+4] = [205, 127, 50]
            else:
                # Healthy - uniform green with texture
                noise = np.random.randint(-20, 20, (224, 224, 3))
                img_array[:, :] = base_green
                img_array = np.clip(img_array.astype(int) + noise, 0, 255).astype(np.uint8)

            img = Image.fromarray(img_array)
            img.save(os.path.join(class_dir, f"image_{i:04d}.jpg"))

    print(f"Sample dataset created at {output_dir}")
    verify_dataset(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PlantVillage dataset")
    parser.add_argument("--source", choices=["kaggle", "huggingface", "sample"],
                        default="kaggle", help="Download source")
    parser.add_argument("--output", default="./data/plantvillage", help="Output directory")
    parser.add_argument("--sample-classes", type=int, default=5,
                        help="Number of classes for sample dataset")
    parser.add_argument("--sample-images", type=int, default=100,
                        help="Images per class for sample dataset")

    args = parser.parse_args()

    if args.source == "kaggle":
        download_via_kaggle(args.output)
    elif args.source == "huggingface":
        download_via_huggingface(args.output)
    elif args.source == "sample":
        create_sample_dataset(args.output, args.sample_classes, args.sample_images)

    verify_dataset(args.output)
