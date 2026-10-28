"""Run both trained models on one new orange image.

The same preprocessing is used here as during training.
"""
import io
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

from .classical import load_model as load_svm
from .cnn import load_model as load_cnn
from .features import features_from_bgr
from .roi import find_fruit_bbox

BASE_DIR = Path(__file__).resolve().parents[1]
Image.MAX_IMAGE_PIXELS = 24_000_000


class ImageValidationError(ValueError):
    pass


@lru_cache(maxsize=1)
def load_models():
    """Load saved models once; reuse them for later uploaded images."""
    folder = BASE_DIR / "models"
    classes = json.loads((folder / "classes.json").read_text(encoding="utf-8"))
    svm = load_svm(folder / "classical_svm.joblib")

    saved = torch.load(folder / "cnn_best.pt", map_location="cpu", weights_only=True)
    cnn = load_cnn(folder / "cnn_best.pt")
    cnn.eval()

    cnn_classes = saved["classes"]
    size = saved["size"]
    if sorted(classes) != sorted(cnn_classes) or sorted(classes) != sorted(svm.classes_):
        raise RuntimeError("The saved models do not have matching class names.")

    image_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    torch.set_num_threads(min(4, torch.get_num_threads()))
    return classes, svm, cnn, cnn_classes, image_transform


def read_uploaded_image(image_bytes):
    """Check the uploaded file before passing it to a model."""
    if not image_bytes:
        raise ImageValidationError("No image was uploaded.")

    try:
        with Image.open(io.BytesIO(image_bytes)) as original:
            if original.format not in {"JPEG", "PNG", "WEBP"}:
                raise ImageValidationError("Only JPG, PNG and WebP images are accepted.")
            width, height = original.size
            if width < 48 or height < 48:
                raise ImageValidationError("Image must be at least 48 x 48 pixels.")
            if width * height > 24_000_000:
                raise ImageValidationError("Image pixel dimensions are too large.")
            image = ImageOps.exif_transpose(original).convert("RGB")
            image.load()
            return image
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        raise ImageValidationError("This file is not a valid image.") from error


def predict_rgb(image):
    classes, svm, cnn, cnn_classes, transform = load_models()

    # Approach 1: OpenCV features + SVM
    rgb_array = np.asarray(image.convert("RGB"))
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    features = features_from_bgr(bgr_array).reshape(1, -1)
    svm_values = svm.predict_proba(features)[0]
    svm_scores = {name: float(value) for name, value in zip(svm.classes_, svm_values)}

    # Approach 2: PyTorch CNN
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = cnn(image_tensor)
        cnn_values = torch.softmax(output, dim=1)[0].numpy()
    cnn_scores = {name: float(value) for name, value in zip(cnn_classes, cnn_values)}

    svm_label = max(svm_scores, key=svm_scores.get)
    cnn_label = max(cnn_scores, key=cnn_scores.get)
    box = find_fruit_bbox(bgr_array)

    return {
        "classes": classes,
        "image_size": {"width": image.width, "height": image.height},
        "svm": {"label": svm_label, "score": svm_scores[svm_label], "scores": svm_scores},
        "cnn": {"label": cnn_label, "score": cnn_scores[cnn_label], "scores": cnn_scores},
        "agreement": svm_label == cnn_label,
        "selected_model": "svm",
        "candidate_bbox_xywh": list(box) if box else None,
        "cautions": [
            "Scores are not calibrated probabilities.",
            "A drawn rectangle is a simple estimate, not object detection.",
            "These labels describe visible conditions, not confirmed diagnoses.",
        ],
    }


def predict_bytes(image_bytes):
    image = read_uploaded_image(image_bytes)
    return predict_rgb(image)
