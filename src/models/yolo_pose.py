from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.models.schemas import ModuleResult, ModuleState
from src.modules.metrics import calculate_metrics, tolerance_failures


class YOLOPoseInspector:
    """YOLO11-pose adapter for Camera 1 dimensions and Camera 2 hook tips."""
    def __init__(self, config: dict) -> None:
        self.config = config
        self.settings = config["yolo_pose"]
        self._models: dict[str, object] = {}

    def inspect_front(self, image: np.ndarray) -> ModuleResult:
        scale = self.config["calibration"].get("px_per_mm")
        result, points, overlay = self._predict(image, "front")
        if result.state != ModuleState.PASS:
            return result.copy(update={"module": 2})
        if not scale:
            return ModuleResult(module=2, state=ModuleState.ERROR, message="Camera 1 px_per_mm calibration is required.")
        try:
            coil_diameter_px = self._coil_diameter_px(image, points)
            dimensions = self.config["dimensions"]
            metrics = calculate_metrics(points, float(scale), dimensions["top_coil_count"], dimensions["bottom_coil_count"], coil_diameter_px)
            failures = tolerance_failures(metrics, self.config)
            return ModuleResult(module=2, state=ModuleState.FAIL if failures else ModuleState.PASS, metrics=metrics,
                details={"keypoints": points, "tolerance_failures": [name for name, _, _ in failures], "overlay": overlay.tolist()})
        except Exception as exc:
            return ModuleResult(module=2, state=ModuleState.ERROR, message=f"Dimension calculation error: {exc}")

    def inspect_top(self, image: np.ndarray) -> ModuleResult:
        result, points, overlay = self._predict(image, "top")
        if result.state != ModuleState.PASS:
            return result.copy(update={"module": 3})
        try:
            center = self._coil_center_from_image(image)
            top_angle = self._angle(center, points["pt1"])
            bottom_angle = self._angle(center, points["pt7"])
            separation = abs((top_angle - bottom_angle + 180) % 360 - 180)
            setting = self.config["module3_hook_angle"]
            error = abs((separation - setting["expected_separation_degrees"] + 180) % 360 - 180)
            state = ModuleState.PASS if error <= setting["max_angle_error_degrees"] else ModuleState.FAIL
            return ModuleResult(module=3, state=state, metrics={"top_hook_angle": top_angle, "bottom_hook_angle": bottom_angle, "hook_separation": separation, "hook_alignment_error": error},
                details={"coil_center": center, "hook_tips": [points["pt1"], points["pt7"]], "threshold": setting["max_angle_error_degrees"], "overlay": overlay.tolist()})
        except Exception as exc:
            return ModuleResult(module=3, state=ModuleState.ERROR, message=f"Hook-angle calculation error: {exc}")

    def _predict(self, image: np.ndarray, view: str) -> tuple[ModuleResult, dict, np.ndarray | None]:
        if not self.settings["enabled"]:
            return ModuleResult(module=2, state=ModuleState.SKIPPED, message="YOLO11-pose disabled; train and configure a checkpoint."), {}, None
        checkpoint = Path(self.settings[f"{view}_checkpoint_path"])
        if not checkpoint.exists():
            return ModuleResult(module=2, state=ModuleState.ERROR, message=f"YOLO pose checkpoint missing: {checkpoint}"), {}, None
        try:
            if view not in self._models:
                from ultralytics import YOLO
                self._models[view] = YOLO(str(checkpoint))
            prediction = self._models[view].predict(image, conf=self.settings["confidence_threshold"], imgsz=self.settings["image_size"], verbose=False)[0]
            if prediction.keypoints is None or len(prediction.keypoints.xy) == 0:
                return ModuleResult(module=2, state=ModuleState.FAIL, message="No spring pose detected."), {}, None
            index = int(prediction.boxes.conf.argmax().item()) if prediction.boxes is not None else 0
            xy = prediction.keypoints.xy[index].cpu().numpy()
            conf = prediction.keypoints.conf[index].cpu().numpy() if prediction.keypoints.conf is not None else np.ones(len(xy))
            names = self.settings["keypoint_names"]
            if len(xy) != len(names):
                raise ValueError(f"Expected {len(names)} YOLO keypoints, received {len(xy)}")
            points = {name: (float(x), float(y)) for name, (x, y), confidence in zip(names, xy, conf) if confidence > 0}
            required = names if view == "front" else ["pt1", "pt7"]
            missing = [name for name in required if name not in points]
            if missing: return ModuleResult(module=2, state=ModuleState.FAIL, message=f"Missing visible keypoints: {missing}"), {}, None
            return ModuleResult(module=2, state=ModuleState.PASS), points, prediction.plot()
        except Exception as exc:
            return ModuleResult(module=2, state=ModuleState.ERROR, message=f"YOLO pose error: {exc}"), {}, None

    @staticmethod
    def _angle(center: tuple[float, float], point: tuple[float, float]) -> float:
        return float(np.degrees(np.arctan2(point[1] - center[1], point[0] - center[0])))

    @staticmethod
    def _coil_center_from_image(image: np.ndarray) -> tuple[float, float]:
        """OpenCV supplies the coil centre; YOLO-pose supplies the two hook tips."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(cv2.GaussianBlur(gray, (5, 5), 0), cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=80, param1=100, param2=35, minRadius=20, maxRadius=min(gray.shape[:2]) // 2)
        if circles is None:
            raise ValueError("OpenCV could not find the coil centre in Camera 2 image.")
        cx, cy, _ = circles[0][0]
        return float(cx), float(cy)

    @staticmethod
    def _coil_diameter_px(image: np.ndarray, points: dict[str, tuple[float, float]]) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        widths = []
        for upper, lower in ((points["pt2"], points["pt3"]), (points["pt5"], points["pt6"])):
            y0, y1 = sorted((max(0, int(upper[1])), min(mask.shape[0], int(lower[1]))))
            if y1 > y0:
                rows = np.count_nonzero(mask[y0:y1], axis=1)
                widths.append(float(np.percentile(rows, 90)))
        if not widths: raise ValueError("Could not measure coil region width.")
        return float(np.mean(widths))
