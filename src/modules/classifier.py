"""
src/modules/classifier.py
Camera 1 inference: image -> YOLO classifier -> GOOD / DEFECTIVE (+ confidence).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

DEFAULT_WEIGHTS = "runs/classify/spring_cls_v4/weights/best.pt"
GOOD_LABELS = {"GOOD", "OK", "PASS"}


class SpringClassifier:
    def __init__(self, weights=DEFAULT_WEIGHTS, imgsz=None, device=None,
                 uncertain_threshold: float = 0.0):
        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(f"Weights not found: {weights}")

        self.model = YOLO(str(weights))
        # Use the size the model was TRAINED at (V1 = 960), unless overridden
        self.imgsz = imgsz or int(self.model.overrides.get("imgsz", 224))
        self.device = device
        self.uncertain_threshold = uncertain_threshold
        self.names = {int(k): str(v).upper() for k, v in self.model.names.items()}
        self.weights = str(weights)
        self._warmup()

    def _warmup(self):
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.model.predict(dummy, imgsz=self.imgsz, device=self.device, verbose=False)

    def predict(self, image) -> dict:
        """image: file path, numpy array (BGR), or PIL image."""
        t0 = time.perf_counter()
        r = self.model.predict(image, imgsz=self.imgsz, device=self.device,
                               verbose=False)[0]
        infer_ms = (time.perf_counter() - t0) * 1000

        probs = r.probs
        label = self.names[int(probs.top1)]
        conf = float(probs.top1conf)

        if conf < self.uncertain_threshold:
            decision = "UNCERTAIN"
        else:
            decision = "ACCEPT" if label in GOOD_LABELS else "REJECT"

        return {
            "prediction": label,
            "confidence": round(conf, 4),
            "decision": decision,
            "probabilities": {self.names[i]: round(float(p), 4)
                              for i, p in enumerate(probs.data.tolist())},
            "inference_ms": round(infer_ms, 1),
        }
