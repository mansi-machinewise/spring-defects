from __future__ import annotations

import cv2
import numpy as np


def preprocess_camera1(image: np.ndarray, size: int = 640) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized silhouette crop and its full-frame foreground mask."""
    resized = cv2.resize(image, (size, size))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No spring contour found in Camera 1 image.")
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    pad = 8
    x0, y0, x1, y1 = max(0, x-pad), max(0, y-pad), min(size, x+w+pad), min(size, y+h+pad)
    crop = resized[y0:y1, x0:x1]
    return crop.astype(np.float32) / 255.0, mask


def preprocess_camera2(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.GaussianBlur(gray, (5, 5), 0)
