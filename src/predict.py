"""
Inference pipeline — load a trained model and run predictions on images.
"""

import io
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from src.dataset import get_inference_transforms
from src.evaluate import predict_single, GradCAM, get_gradcam_layer
from src.model import load_model


class CropDiseasePredictor:
    """
    High-level predictor class.
    Handles image loading, preprocessing, inference, and optional Grad-CAM.
    """

    # Human-readable disease information
    DISEASE_INFO = {
        "Apple___Apple_scab": {
            "common_name": "Apple Scab",
            "description": "Fungal disease causing dark, scaly lesions on leaves and fruit.",
            "treatment": "Apply fungicides (myclobutanil, captan). Remove and destroy infected leaves.",
            "severity": "moderate",
        },
        "Apple___Black_rot": {
            "common_name": "Apple Black Rot",
            "description": "Fungal infection causing dark rotting spots on fruit and frogeye spots on leaves.",
            "treatment": "Prune dead wood. Apply copper-based fungicides. Remove mummified fruit.",
            "severity": "high",
        },
        "Apple___Cedar_apple_rust": {
            "common_name": "Cedar Apple Rust",
            "description": "Fungal disease with bright orange leaf spots, requiring two host plants.",
            "treatment": "Apply myclobutanil fungicides in spring. Remove nearby cedar/juniper hosts.",
            "severity": "moderate",
        },
        "Tomato___Early_blight": {
            "common_name": "Tomato Early Blight",
            "description": "Fungal disease causing dark concentric ring spots on lower leaves.",
            "treatment": "Apply chlorothalonil or copper fungicide. Remove affected leaves. Mulch soil.",
            "severity": "moderate",
        },
        "Tomato___Late_blight": {
            "common_name": "Tomato Late Blight",
            "description": "Oomycete causing water-soaked lesions that spread rapidly in cool, wet conditions.",
            "treatment": "Apply mancozeb or chlorothalonil. Destroy infected plants. Improve air circulation.",
            "severity": "high",
        },
        "Tomato___Leaf_Mold": {
            "common_name": "Tomato Leaf Mold",
            "description": "Fungal disease with pale yellow patches on leaf tops and olive mold underneath.",
            "treatment": "Improve ventilation. Reduce humidity. Apply copper-based fungicides.",
            "severity": "moderate",
        },
        "Tomato___Septoria_leaf_spot": {
            "common_name": "Septoria Leaf Spot",
            "description": "Fungal disease causing small circular spots with dark borders on lower leaves.",
            "treatment": "Remove infected leaves. Apply chlorothalonil or copper fungicide.",
            "severity": "moderate",
        },
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
            "common_name": "Tomato Yellow Leaf Curl Virus",
            "description": "Virus transmitted by whiteflies causing upward leaf curling and yellowing.",
            "treatment": "Control whitefly populations. Use resistant varieties. Remove infected plants.",
            "severity": "high",
        },
        "Potato___Early_blight": {
            "common_name": "Potato Early Blight",
            "description": "Fungal disease causing dark target-like rings on older leaves.",
            "treatment": "Apply azoxystrobin or chlorothalonil. Ensure proper soil nutrition.",
            "severity": "moderate",
        },
        "Potato___Late_blight": {
            "common_name": "Potato Late Blight",
            "description": "Caused by Phytophthora infestans; historic cause of Irish Potato Famine.",
            "treatment": "Apply mancozeb preventively. Destroy infected crops. Use certified seed potatoes.",
            "severity": "critical",
        },
        "Corn_(maize)___Common_rust_": {
            "common_name": "Corn Common Rust",
            "description": "Fungal disease with small, round, brown pustules on both leaf surfaces.",
            "treatment": "Apply triazole fungicides. Plant resistant hybrids.",
            "severity": "low",
        },
        "Corn_(maize)___Northern_Leaf_Blight": {
            "common_name": "Northern Corn Leaf Blight",
            "description": "Fungal disease causing long, cigar-shaped gray-green lesions.",
            "treatment": "Apply fungicides at early onset. Plant resistant hybrids. Rotate crops.",
            "severity": "moderate",
        },
        "Grape___Black_rot": {
            "common_name": "Grape Black Rot",
            "description": "Fungal disease causing brown leaf spots and shriveled mummified berries.",
            "treatment": "Apply myclobutanil or captan before bloom. Remove mummies.",
            "severity": "high",
        },
    }

    def __init__(self, checkpoint_path: str, device: str = "auto"):
        self.model, self.class_to_idx, self.metadata = load_model(checkpoint_path, device)
        # Resolve device from actual model parameter placement
        self.device = next(self.model.parameters()).device.type
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.transform = get_inference_transforms(224)
        self._gradcam = None

    def _enable_gradcam(self):
        if self._gradcam is None:
            layer = get_gradcam_layer(self.model)
            if layer is not None:
                self._gradcam = GradCAM(self.model, layer)

    def predict(
        self,
        image: Union[str, Path, bytes, Image.Image, np.ndarray],
        top_k: int = 5,
        include_info: bool = True,
    ) -> Dict:
        """
        Run inference on an image.

        Args:
            image: File path, bytes, PIL Image, or numpy array.
            top_k: Number of top predictions to return.
            include_info: Whether to include disease treatment info.

        Returns:
            Dict with predictions, top class details, and optional disease info.
        """
        pil_image = self._load_image(image)
        tensor = self._preprocess(pil_image)
        predictions = predict_single(self.model, tensor, self.idx_to_class, self.device, top_k)

        result = {
            "predictions": predictions,
            "top_prediction": predictions[0],
            "image_size": pil_image.size,
        }

        if include_info:
            top_class = predictions[0]["class_name"]
            result["disease_info"] = self.DISEASE_INFO.get(top_class, self._default_info(top_class))

        return result

    def _load_image(self, image) -> Image.Image:
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        elif isinstance(image, bytes):
            return Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            return image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def _preprocess(self, pil_image: Image.Image) -> torch.Tensor:
        img_array = np.array(pil_image)
        augmented = self.transform(image=img_array)
        return augmented["image"]

    @staticmethod
    def _default_info(class_name: str) -> Dict:
        parts = class_name.split("___")
        plant = parts[0].replace("_", " ").replace("(", "").replace(")", "").strip()
        condition = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
        is_healthy = "healthy" in class_name.lower()
        return {
            "common_name": condition,
            "description": (
                f"Healthy {plant} plant." if is_healthy
                else f"{condition} detected on {plant}."
            ),
            "treatment": (
                "No treatment needed." if is_healthy
                else "Consult local agricultural extension for treatment recommendations."
            ),
            "severity": "none" if is_healthy else "unknown",
        }

    def format_class_name(self, class_name: str) -> Dict[str, str]:
        """Format raw class name into human-readable parts."""
        parts = class_name.split("___")
        plant = parts[0].replace("_", " ").title()
        condition = parts[1].replace("_", " ").title() if len(parts) > 1 else "Unknown"
        return {"plant": plant, "condition": condition, "raw": class_name}
