"""
FastAPI application for Crop Disease Classifier.
Endpoints:
  POST /predict          — upload an image, get disease predictions
  GET  /health           — health check
  GET  /classes          — list all supported classes
  GET  /                 — serve frontend
"""

import io
import os
import sys
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import uvicorn

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.predict import CropDiseasePredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Crop Disease Classifier API",
    description="Detect diseases in crop leaf images using EfficientNet transfer learning.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.mount("/static", StaticFiles(directory=str(ROOT / "app" / "static")), name="static")
templates = Jinja2Templates(directory=str(ROOT / "app" / "templates"))

# ── Model loading ─────────────────────────────────────────────────────────────
CHECKPOINT_PATH = os.environ.get("MODEL_PATH", str(ROOT / "models" / "best_model.pth"))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "10"))

predictor: Optional[CropDiseasePredictor] = None


@app.on_event("startup")
async def load_model():
    global predictor
    if Path(CHECKPOINT_PATH).exists():
        logger.info(f"Loading model from {CHECKPOINT_PATH}...")
        predictor = CropDiseasePredictor(CHECKPOINT_PATH)
        logger.info("Model loaded successfully.")
    else:
        logger.warning(
            f"Model checkpoint not found at {CHECKPOINT_PATH}. "
            "The /predict endpoint will return a demo response. "
            "Train a model first with: python src/train.py --data ./data/plantvillage"
        )


# ── Helper ────────────────────────────────────────────────────────────────────

def check_image(file: UploadFile) -> None:
    allowed = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=400, detail=f"File must be an image (jpeg/png/webp). Got: {file.content_type}")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
@app.head("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "checkpoint": CHECKPOINT_PATH,
    }


@app.get("/classes")
async def get_classes():
    if predictor is None:
        # Return config classes if model not loaded
        import yaml
        with open(ROOT / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        classes = cfg.get("classes", [])
    else:
        classes = list(predictor.class_to_idx.keys())

    formatted = []
    for cls in classes:
        parts = cls.split("___")
        formatted.append({
            "raw": cls,
            "plant": parts[0].replace("_", " ").title(),
            "condition": parts[1].replace("_", " ").title() if len(parts) > 1 else "Unknown",
        })

    return {"total": len(classes), "classes": formatted}


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 5):
    check_image(file)

    # Check file size
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_UPLOAD_MB}MB.")

    # Validate image
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    # Run prediction (or demo response if model not loaded)
    if predictor is None:
        return _demo_response(img)

    try:
        result = predictor.predict(img, top_k=min(top_k, 10), include_info=True)
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed. Check server logs.")

    top = result["top_prediction"]
    plant_info = predictor.format_class_name(top["class_name"])

    return {
        "status": "success",
        "top_prediction": {
            "class_name": top["class_name"],
            "plant": plant_info["plant"],
            "condition": plant_info["condition"],
            "confidence": round(top["probability"] * 100, 2),
            "is_healthy": "healthy" in top["class_name"].lower(),
        },
        "all_predictions": [
            {
                "rank": i + 1,
                "class_name": p["class_name"],
                "plant": predictor.format_class_name(p["class_name"])["plant"],
                "condition": predictor.format_class_name(p["class_name"])["condition"],
                "confidence": round(p["probability"] * 100, 2),
            }
            for i, p in enumerate(result["predictions"])
        ],
        "disease_info": result.get("disease_info", {}),
        "image_size": result["image_size"],
    }


def _demo_response(img: Image.Image) -> JSONResponse:
    """Return a demo response when model is not loaded."""
    return JSONResponse({
        "status": "demo",
        "message": "Model not loaded. This is a demo response. Train the model first.",
        "top_prediction": {
            "class_name": "Tomato___Late_blight",
            "plant": "Tomato",
            "condition": "Late Blight",
            "confidence": 94.3,
            "is_healthy": False,
        },
        "all_predictions": [
            {"rank": 1, "class_name": "Tomato___Late_blight", "plant": "Tomato", "condition": "Late Blight", "confidence": 94.3},
            {"rank": 2, "class_name": "Tomato___Early_blight", "plant": "Tomato", "condition": "Early Blight", "confidence": 3.2},
            {"rank": 3, "class_name": "Potato___Late_blight", "plant": "Potato", "condition": "Late Blight", "confidence": 1.5},
        ],
        "disease_info": {
            "common_name": "Tomato Late Blight",
            "description": "Caused by Phytophthora infestans; historic cause of Irish Potato Famine.",
            "treatment": "Apply mancozeb preventively. Destroy infected crops.",
            "severity": "high",
        },
        "image_size": list(img.size),
    })


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        reload=os.environ.get("ENV", "production") == "development",
        workers=1,
    )
