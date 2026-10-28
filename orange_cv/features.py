"""Color and texture features used by the SVM model."""
import cv2
import numpy as np
from skimage.feature import local_binary_pattern

from .data import read_image


def features_from_bgr(image):
    """Convert one image into 157 numerical features."""
    image = cv2.resize(image, (160, 160), interpolation=cv2.INTER_AREA)

    # Most training photos have one orange near the middle of the picture.
    # Using the middle oval reduces (but does not remove) the background.
    mask = np.zeros((160, 160), dtype=np.uint8)
    cv2.ellipse(mask, (80, 80), (66, 72), 0, 0, 360, 255, -1)

    features = []

    # HSV represents hue/saturation, and LAB represents color differences.
    for color_code in [cv2.COLOR_BGR2HSV, cv2.COLOR_BGR2LAB]:
        converted = cv2.cvtColor(image, color_code)
        for channel in range(3):
            histogram = cv2.calcHist([converted], [channel], mask, [24], [0, 256])
            histogram = histogram.flatten()
            histogram = histogram / (histogram.sum() + 1e-8)
            features.extend(histogram)

    # Local Binary Patterns: simple measurement of peel texture.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp[mask > 0], bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype(np.float32)
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-8)
    features.extend(lbp_hist)

    # Three extra image statistics: brightness, contrast and sharpness.
    middle = gray[mask > 0].astype(np.float32)
    features.append(middle.mean() / 255.0)
    features.append(middle.std() / 255.0)
    features.append(cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0)

    return np.array(features, dtype=np.float32)


def features_from_path(image_path):
    return features_from_bgr(read_image(image_path))
