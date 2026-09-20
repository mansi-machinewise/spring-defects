from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np


class CameraCaptureService:
    """Hardware adapter. Replace `capture_pair` for PLC/GPIO SDK triggering."""
    def __init__(self, config: dict) -> None:
        self.config = config

    def capture_pair(self, camera1_path: str | None = None, camera2_path: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        if camera1_path and camera2_path:
            return self._read(camera1_path), self._read(camera2_path)
        if self.config["camera"]["mode"] == "simulation":
            return self._simulated_front(), self._simulated_top()
        with ThreadPoolExecutor(max_workers=2) as pool:
            a = pool.submit(self._capture_one, self.config["camera"]["camera1_index"])
            b = pool.submit(self._capture_one, self.config["camera"]["camera2_index"])
            return a.result(), b.result()

    @staticmethod
    def _read(path: str) -> np.ndarray:
        image = cv2.imread(str(Path(path)))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return image

    @staticmethod
    def _capture_one(index: int) -> np.ndarray:
        cap = cv2.VideoCapture(index)
        try:
            ok, image = cap.read()
            if not ok: raise RuntimeError(f"Camera {index} capture failed")
            return image
        finally:
            cap.release()

    @staticmethod
    def _simulated_front() -> np.ndarray:
        image = np.full((640, 640, 3), 245, dtype=np.uint8)
        cv2.line(image, (320, 170), (320, 470), (20, 20, 20), 8)
        for y in (145, 175, 465, 495): cv2.ellipse(image, (320, y), (65, 18), 0, 0, 360, (20, 20, 20), 7)
        cv2.line(image, (255, 145), (220, 105), (20, 20, 20), 8); cv2.line(image, (385, 495), (420, 535), (20, 20, 20), 8)
        return image

    @staticmethod
    def _simulated_top() -> np.ndarray:
        image = np.full((640, 640, 3), 245, dtype=np.uint8)
        cv2.circle(image, (320, 320), 130, (20, 20, 20), 9)
        # Nominal spring: hook tips are in opposing angular directions.
        cv2.line(image, (320, 190), (320, 95), (20, 20, 20), 9)
        cv2.line(image, (320, 450), (320, 545), (20, 20, 20), 9)
        return image
