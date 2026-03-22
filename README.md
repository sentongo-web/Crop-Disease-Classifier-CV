# Crop Disease Classifier

This project is a fully working crop disease detection system built with deep learning. You upload a photo of a plant leaf, and the app tells you what disease it has, how severe it is, and what treatment to apply. It runs entirely on CPU — no GPU needed.

The classifier recognises **38 disease classes across 14 crops**, including Apple, Tomato, Potato, Corn, Grape, Peach, Pepper, Cherry, Blueberry, Orange, Raspberry, Soybean, Squash and Strawberry. The dataset comes from PlantVillage, which contains over 54,000 annotated leaf images.

---

## How It Works

The system has three layers:

1. **A deep learning model** trained with transfer learning on top of a pre-trained MobileNetV3 backbone. The model learns to distinguish between healthy leaves and diseased ones by fine-tuning weights that were originally trained on ImageNet.

2. **A prediction pipeline** that accepts any image (file path, bytes, PIL Image, or NumPy array), pre-processes it, runs it through the model, and returns the top predictions along with disease treatment info.

3. **A FastAPI web application** that wraps the pipeline behind a REST API and serves a drag-and-drop web interface where anyone can upload a leaf photo and get a diagnosis in seconds.

---

## Project Structure

Here is what each file and folder does:

```text
Crop-Disease-Classifier-CV/
│
├── src/                         Core machine learning code
│   ├── dataset.py               Loads images, applies augmentations, creates DataLoaders
│   ├── model.py                 Defines the neural network architecture
│   ├── train.py                 Runs the training loop, saves checkpoints
│   ├── evaluate.py              Computes accuracy, F1, confusion matrix, Grad-CAM
│   └── predict.py               High-level class for running inference on new images
│
├── app/                         Web application
│   ├── main.py                  FastAPI routes and startup logic
│   ├── templates/index.html     HTML frontend (drag-and-drop UI)
│   └── static/                  CSS styles and JavaScript
│
├── data/
│   └── download_dataset.py      Downloads PlantVillage from Kaggle/HuggingFace,
│                                or generates a synthetic sample dataset
│
├── scripts/
│   ├── run_local.py             Starts the API server locally
│   └── deploy.sh                Builds and runs the Docker container
│
├── notebooks/
│   └── train_and_evaluate.ipynb End-to-end Jupyter walkthrough
│
├── models/                      Saved model checkpoints go here (git-ignored)
├── config.yaml                  All training hyperparameters and class names
├── requirements.txt             Python dependencies (CPU-only PyTorch)
├── setup_env.bat                One-click Windows setup script
├── Dockerfile                   Multi-stage Docker build
└── docker-compose.yml           Docker Compose with optional Nginx
```

---

## Getting Started

### Step 1 — Run the setup script (Windows)

```bat
setup_env.bat
```

This script creates a Python virtual environment in the `venv/` folder and installs all dependencies from `requirements.txt`. PyTorch is downloaded as a CPU-only build (~250 MB), so you do not need a GPU or CUDA.

After it finishes, activate the environment manually for future terminal sessions:

```bat
venv\Scripts\activate
```

### Step 2 — Create a dataset

You have three options depending on what you want to do.

**Option A — Quick synthetic dataset (no internet needed, good for testing)**

```bash
python data\download_dataset.py --source sample --output data\sample --sample-classes 5 --sample-images 100
```

This generates 500 synthetic leaf images across 5 disease classes. The images are not real photographs, but they are enough to verify the entire training pipeline works end to end.

**Option B — Full PlantVillage dataset via HuggingFace (real data, ~800 MB)**

```bash
python data\download_dataset.py --source huggingface --output data\plantvillage
```

This downloads all 54,000+ real plant images. Training on this will give you a genuinely accurate model.

**Option C — Kaggle (requires a Kaggle account and `~/.kaggle/kaggle.json`)**

```bash
python data\download_dataset.py --source kaggle --output data\plantvillage
```

After downloading, your data folder should look like this:

```text
data/plantvillage/
    Apple___Apple_scab/
        image_0001.jpg
        image_0002.jpg
    Apple___healthy/
        ...
    Tomato___Late_blight/
        ...
```

Each subfolder name is the class label. The dataset code discovers classes automatically by reading this folder structure.

### Step 3 — Train the model

```bash
# Quick test on synthetic data (finishes in a few minutes on CPU)
python src\train.py --data data\sample --arch mobilenet_v3_small --epochs 5 --batch-size 8

# Full training on real PlantVillage data (takes several hours on CPU)
python src\train.py --data data\plantvillage --arch mobilenet_v3_small --epochs 15 --batch-size 16
```

The training script prints a progress line after each epoch:

```text
Epoch [  1/15] train_loss=1.2345 train_acc=0.6123 val_loss=0.9876 val_acc=0.7210 lr=1.00e-03 (45.2s)
  [SAVED] Best model (val_acc=0.7210)
```

When it finishes, the `models/` folder will contain:

- `best_model.pth` — the checkpoint with the highest validation accuracy
- `last_model.pth` — the final epoch checkpoint
- `training_history.json` — loss and accuracy for every epoch
- `test_metrics.json` — accuracy, F1 and per-class breakdown on the held-out test set

### Step 4 — Start the web app

```bash
python scripts\run_local.py
```

Open your browser at `http://localhost:8000`. You will see a drag-and-drop interface where you can upload any leaf photo. The app calls the `/predict` endpoint, displays the top prediction with a confidence percentage, shows all alternative predictions ranked by confidence, and provides disease description, treatment recommendations and severity level.

If no trained model is found, the app starts in demo mode with a mock response so you can explore the UI immediately.

---

## How the Code Was Built

### Dataset loading — `src/dataset.py`

The dataset module handles everything between raw image files on disk and ready-to-train PyTorch DataLoaders.

The first thing it does is discover all class folders inside the data directory and assign each one a numeric index. This is stored in a `class_to_idx` dictionary, for example `{"Apple___Apple_scab": 0, "Apple___healthy": 1, ...}`. This mapping is saved inside every model checkpoint so predictions can be decoded back to human-readable names without needing the original data folder.

Images are loaded with Pillow and converted to NumPy arrays before being passed to Albumentations, which is the augmentation library. During training, each image goes through a randomised pipeline that includes random cropping and resizing, horizontal flipping, rotation up to 30 degrees, random blur, colour jitter, CLAHE contrast enhancement, and coarse dropout (random rectangular patches of pixels are blacked out). This variety means the model never sees the exact same image twice and learns to be robust to camera angle, lighting and focus variations.

During validation and inference, no random augmentation is applied — images are just resized to 224x224 and normalised using the ImageNet mean and standard deviation.

The DataLoader uses a `WeightedRandomSampler` to handle class imbalance. Some disease classes in PlantVillage have far more images than others. Without rebalancing, the model would learn to predict the common classes well and ignore rare ones. The sampler gives rarer classes a higher probability of being drawn in each batch so every class gets roughly equal representation during training.

On Windows, using multiple DataLoader worker processes causes problems due to how Python's multiprocessing spawns processes on that platform. The `num_workers` parameter is automatically set to 0 on Windows, which runs all data loading in the main process. This is slightly slower but completely reliable.

### Model architecture — `src/model.py`

The model is built using transfer learning. Instead of training a neural network from scratch (which would require millions of images and days of compute), we start from a network that was already trained on ImageNet's 1.2 million images. The pre-trained weights capture general visual features like edges, textures and shapes. We then fine-tune those weights to specialise on plant disease patterns.

The `CropDiseaseClassifier` class wraps any of six supported backbone architectures:

- `mobilenet_v3_small` — the default, around 2.5 million parameters, fast on CPU
- `mobilenet_v3_large` — slightly larger and more accurate
- `efficientnet_b0` and `efficientnet_b3` — a more modern family with excellent accuracy-to-size ratios
- `resnet34` and `resnet50` — classic architectures that are well-understood and reliable

Regardless of which backbone you choose, the code removes the original classification head and replaces it with a custom sequence: Dropout → Linear(512) → BatchNorm → ReLU → Dropout → Linear(num_classes). This head is what gets trained to distinguish crop diseases. The backbone is kept frozen initially so only the head trains, then the backbone is unfrozen for fine-tuning with a much lower learning rate.

Loading a saved checkpoint uses `weights_only=False` because the checkpoint also stores the class mapping and metadata, not just the network weights.

### Training loop — `src/train.py`

Training happens in two phases.

In Phase 1, the backbone is frozen and only the new classification head trains. This runs for a configurable number of warm-up epochs (default 3). The idea is to train the head to a reasonable starting point before touching the pre-trained backbone weights, which prevents the backbone from being destabilised early in training when the head is still outputting near-random predictions.

In Phase 2, the backbone is unfrozen and the whole network trains together. The backbone gets a learning rate ten times lower than the head (0.0001 versus 0.001) because we want to make small adjustments to the pre-trained features rather than overwrite them completely. This technique is called differential learning rates.

The learning rate follows a cosine annealing schedule, which smoothly decreases from the starting value to near zero over the course of training. This tends to produce better final accuracy than a fixed learning rate.

The loss function is cross-entropy with label smoothing of 0.1. Label smoothing makes the model less overconfident by targeting a distribution of 0.9 for the correct class and spreading 0.1 across all others, which improves generalisation.

Early stopping watches the validation loss and stops training automatically if it has not improved for 7 consecutive epochs, which saves time and prevents overfitting.

### Evaluation — `src/evaluate.py`

After training, the best checkpoint is evaluated on the held-out test set (the 10% of images that were never seen during training or validation).

The evaluation computes overall accuracy, macro F1, weighted F1, precision, recall, and a full per-class classification report. All of this is saved to `models/test_metrics.json`.

The module also implements Grad-CAM (Gradient-weighted Class Activation Mapping), which is an explainability technique. It highlights which parts of a leaf image the model paid the most attention to when making a prediction. This is useful for verifying that the model is looking at disease lesions and spots rather than background soil or labels.

### Inference pipeline — `src/predict.py`

The `CropDiseasePredictor` class provides a single `.predict()` method that accepts an image in any format — file path, raw bytes, PIL Image, or NumPy array — and returns a structured result dictionary with the top-k class predictions, confidence scores, and disease information.

For diseases that appear in the built-in `DISEASE_INFO` dictionary, the result includes a human-readable common name, a description of what causes the disease and what it looks like, treatment recommendations, and a severity level from `none` (healthy) through `low`, `moderate`, `high`, to `critical`. For any class not in that dictionary, a sensible default is generated from the class name itself.

### Web application — `app/main.py`

The API is built with FastAPI, which is a modern Python web framework that automatically generates interactive API documentation at `/docs`.

There are four endpoints. `GET /` serves the web frontend. `GET /health` returns whether the model is loaded and ready. `GET /classes` returns all 38 disease classes with their human-readable plant and condition names. `POST /predict` accepts an image file upload, validates its type and size, runs it through the predictor, and returns the full result as JSON.

If the model checkpoint is not found when the server starts, the API enters demo mode. Every prediction request returns a realistic mock response so the UI works and you can explore the interface without having trained a model first.

---

## Training Arguments Reference

| Argument | Default | What it does |
| --- | --- | --- |
| `--data` | required | Path to your dataset folder |
| `--arch` | `mobilenet_v3_small` | Which backbone to use |
| `--epochs` | `10` | How many full passes through the training data |
| `--batch-size` | `16` | Number of images processed in each step |
| `--lr` | `0.001` | Starting learning rate |
| `--freeze-epochs` | `3` | How many warm-up epochs before unfreezing the backbone |
| `--scheduler` | `cosine` | How the learning rate changes over time |
| `--patience` | `7` | Epochs without improvement before stopping early |
| `--num-workers` | `0` | DataLoader worker processes (keep at 0 on Windows) |
| `--save-dir` | `./models` | Where checkpoints and results are written |

---

## API Usage

Once the server is running, you can call it from the command line:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/leaf.jpg" \
  -F "top_k=5"
```

Or from Python:

```python
import requests

with open("leaf.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f},
        data={"top_k": 5},
    )

result = response.json()
print(result["top_prediction"]["condition"])   # e.g. "Late Blight"
print(result["top_prediction"]["confidence"])  # e.g. 94.3
print(result["disease_info"]["treatment"])     # treatment recommendation
```

A successful response looks like this:

```json
{
  "status": "success",
  "top_prediction": {
    "class_name": "Tomato___Late_blight",
    "plant": "Tomato",
    "condition": "Late Blight",
    "confidence": 94.3,
    "is_healthy": false
  },
  "all_predictions": [
    {"rank": 1, "class_name": "Tomato___Late_blight", "confidence": 94.3},
    {"rank": 2, "class_name": "Tomato___Early_blight", "confidence": 3.2},
    {"rank": 3, "class_name": "Potato___Late_blight", "confidence": 1.5}
  ],
  "disease_info": {
    "common_name": "Tomato Late Blight",
    "description": "Oomycete causing water-soaked lesions that spread rapidly in cool, wet conditions.",
    "treatment": "Apply mancozeb preventively. Destroy infected plants. Improve air circulation.",
    "severity": "high"
  },
  "image_size": [1024, 768]
}
```

---

## Docker

To run the app inside Docker without installing anything on your machine:

```bash
docker compose up --build
```

The Dockerfile uses a multi-stage build. The first stage installs all build dependencies and packages. The second stage copies only what is needed to run the app, keeping the final image small. PyTorch is installed with the CPU-only flag so the image does not balloon to several gigabytes.

For production with HTTPS and static file caching, add the Nginx reverse proxy:

```bash
docker compose --profile production up --build
```

---

## System Requirements

- Python 3.10 or newer
- No GPU or CUDA needed
- About 2 GB of free disk space for dependencies
- At least 4 GB of RAM for training

---

## License

MIT
