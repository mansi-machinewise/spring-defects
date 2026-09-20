from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4
import cv2
import numpy as np

from src.camera.capture import CameraCaptureService
from src.decision.engine import DecisionEngine
from src.models.database import InspectionRepository
from src.models.schemas import InspectRequest, InspectionResult, ModuleResult, ModuleState
from src.models.yolo_detector import YOLOSpringDetector
from src.models.yolo_pose import YOLOPoseInspector
from src.modules.patchcore import PatchCoreInspector
from src.preprocessing.preprocess import preprocess_camera1
from src.reports.report_generator import DefectReportGenerator
from src.video.frame_capture import BestFrameSelector


class InspectionService:
    """Application service (controller-independent orchestration)."""
    def __init__(self, config: dict) -> None:
        self.config = config
        self.capture = CameraCaptureService(config)
        self.detector = YOLOSpringDetector(config)
        self.frame_selector = BestFrameSelector(config, self.detector)
        self.patchcore = PatchCoreInspector(config)
        self.pose = YOLOPoseInspector(config)
        self.engine = DecisionEngine(config)
        self.repo = InspectionRepository()
        self.reports = DefectReportGenerator()

    def inspect(self, request: InspectRequest) -> InspectionResult:
        detection_result: ModuleResult
        if request.camera1_video_path and request.camera2_video_path:
            first = self.frame_selector.select(request.camera1_video_path)
            second = self.frame_selector.select(request.camera2_video_path)
            camera1, camera2 = first.frame, second.frame
            detection_result = self._combine_detection_results(first.detection_result, second.detection_result)
        else:
            camera1, camera2 = self.capture.capture_pair(request.camera1_path, request.camera2_path)
            first, _ = self.detector.detect(camera1)
            second, _ = self.detector.detect(camera2)
            detection_result = self._combine_detection_results(first, second)
        prepared1, _ = preprocess_camera1(camera1)
        with ThreadPoolExecutor(max_workers=self.config["inspection"]["max_parallel_workers"]) as pool:
            one = pool.submit(self.patchcore.inspect, prepared1)
            two = pool.submit(self.pose.inspect_front, camera1)
            three = pool.submit(self.pose.inspect_top, camera2)
            module_results = [detection_result, one.result(), two.result(), three.result()]
        inspection_id = uuid4().hex
        artifact_dir = Path("reports") / inspection_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        camera1_path, camera2_path = artifact_dir / "camera1.png", artifact_dir / "camera2.png"
        cv2.imwrite(str(camera1_path), camera1); cv2.imwrite(str(camera2_path), camera2)
        module_by_id = {result.module: result for result in module_results}
        module_by_id[1].artifacts["camera1_source"] = str(camera1_path)
        module_by_id[3].artifacts["camera2_source"] = str(camera2_path)
        self._save_pose_overlay(module_by_id[2], artifact_dir, "camera1_keypoints.png")
        self._save_anomaly_heatmap(module_by_id[1], artifact_dir)
        self._save_pose_overlay(module_by_id[3], artifact_dir, "camera2_hooks.png")
        decision, failures = self.engine.decide(module_results)
        result = InspectionResult(inspection_id=inspection_id, timestamp=datetime.now(timezone.utc),
            spring_id=request.spring_id, batch_id=request.batch_id, decision=decision,
            failures=failures, module_results=module_results)
        result.report_path = self.reports.generate(result)
        self.repo.save(result, json.dumps(result.model_dump(mode="json")))
        return result

    @staticmethod
    def _save_pose_overlay(result, folder: Path, filename: str) -> None:
        overlay = result.details.pop("overlay", None)
        if overlay is not None:
            path = folder / filename
            cv2.imwrite(str(path), np.asarray(overlay, dtype=np.uint8))
            result.artifacts["keypoint_overlay"] = str(path)

    @staticmethod
    def _save_anomaly_heatmap(result, folder: Path) -> None:
        anomaly_map = result.details.pop("anomaly_map", None)
        if anomaly_map is not None:
            heatmap = np.asarray(anomaly_map, dtype=np.float32)
            heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            path = folder / "anomaly_heatmap.png"
            cv2.imwrite(str(path), cv2.applyColorMap(heatmap, cv2.COLORMAP_JET))
            result.artifacts["anomaly_heatmap"] = str(path)

    @staticmethod
    def _combine_detection_results(first: ModuleResult, second: ModuleResult) -> ModuleResult:
        if first.state == ModuleState.SKIPPED and second.state == ModuleState.SKIPPED:
            return ModuleResult(module=0, state=ModuleState.SKIPPED, message="YOLO spring detector disabled.")
        if first.state == ModuleState.PASS and second.state == ModuleState.PASS:
            return ModuleResult(module=0, state=ModuleState.PASS, metrics={"camera1_detection_confidence": first.metrics["detection_confidence"], "camera2_detection_confidence": second.metrics["detection_confidence"]})
        messages = "; ".join(filter(None, [first.message, second.message]))
        state = ModuleState.ERROR if ModuleState.ERROR in (first.state, second.state) else ModuleState.FAIL
        return ModuleResult(module=0, state=state, message=messages or "Spring was not confirmed in both camera feeds.")
