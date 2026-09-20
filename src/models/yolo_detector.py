from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np

from src.models.schemas import ModuleResult, ModuleState


class SpringDetection(NamedTuple):
    confidence: float
    bbox: tuple[int, int, int, int]


class YOLOSpringDetector:
    """YOLO11 detection adapter used to confirm spring presence before inspection."""
    def __init__(self, config: dict) -> None:
        self.settings = config["yolo_detect"]
        self._model = None

    @property
    def enabled(self) -> bool:
        return bool(self.settings["enabled"])

    def detect(self, image: np.ndarray) -> tuple[ModuleResult, SpringDetection | None]:
        if not self.enabled:
            return ModuleResult(module=0, state=ModuleState.SKIPPED, message="YOLO spring detector disabled."), None
        checkpoint = Path(self.settings["checkpoint_path"])
        if not checkpoint.exists():
            return ModuleResult(module=0, state=ModuleState.ERROR, message=f"YOLO detect checkpoint missing: {checkpoint}"), None
        try:
            if self._model is None:
                from ultralytics import YOLO
                self._model = YOLO(str(checkpoint))
            result = self._model.predict(image, conf=self.settings["confidence_threshold"], imgsz=self.settings["image_size"], verbose=False)[0]
            candidates = []
            for box in result.boxes:
                if int(box.cls.item()) != self.settings["spring_class_id"]:
                    continue
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                candidates.append(SpringDetection(float(box.conf.item()), (x1, y1, x2, y2)))
            if not candidates:
                return ModuleResult(module=0, state=ModuleState.FAIL, message="No spring detected in frame."), None
            selected = max(candidates, key=lambda item: item.confidence)
            return ModuleResult(module=0, state=ModuleState.PASS, metrics={"detection_confidence": selected.confidence}, details={"bbox": selected.bbox}), selected
        except Exception as exc:
            return ModuleResult(module=0, state=ModuleState.ERROR, message=f"YOLO detection error: {exc}"), None
