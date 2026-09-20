from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

from src.models.schemas import ModuleResult, ModuleState


@dataclass
class SelectedFrame:
    frame: np.ndarray
    sharpness: float
    detection_result: ModuleResult


class BestFrameSelector:
    """Select the sharpest spring-confirmed frame from the 14–15 s production window."""
    def __init__(self, config: dict, detector) -> None:
        self.settings = config["camera"]["frame_selection"]
        self.detector = detector

    def select(self, source: str | int) -> SelectedFrame:
        capture = cv2.VideoCapture(str(source) if isinstance(source, Path) else source)
        try:
            if not capture.isOpened(): raise RuntimeError(f"Could not open video source: {source}")
            fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            start, end = self.settings["manufacture_window_start_seconds"], self.settings["manufacture_window_end_seconds"]
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
            best: SelectedFrame | None = None
            while capture.get(cv2.CAP_PROP_POS_FRAMES) <= int(end * fps):
                ok, frame = capture.read()
                if not ok: break
                detection, _ = self.detector.detect(frame)
                if detection.state not in (ModuleState.PASS, ModuleState.SKIPPED):
                    continue
                sharpness = float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
                if sharpness < self.settings["minimum_sharpness"]: continue
                if best is None or sharpness > best.sharpness:
                    best = SelectedFrame(frame, sharpness, detection)
            if best is None: raise RuntimeError("No spring-confirmed sharp frame found in the 14–15 second window.")
            return best
        finally:
            capture.release()
