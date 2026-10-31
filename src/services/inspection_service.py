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
from src.models.schemas import (
    InspectRequest,
    InspectionResult,
    ModuleResult,
    ModuleState,
)
from src.models.yolo_detector import YOLOSpringDetector
from src.models.yolo_pose import YOLOPoseInspector
from src.modules.patchcore import PatchCoreInspector
from src.reports.report_generator import DefectReportGenerator
from src.video.frame_capture import BestFrameSelector


class InspectionService:
    """Application service (controller-independent orchestration)."""

    def __init__(self, config: dict) -> None:
        self.config = config

        self.capture = CameraCaptureService(config)
        self.detector = YOLOSpringDetector(config)
        self.frame_selector = BestFrameSelector(
            config,
            self.detector,
        )
        self.patchcore = PatchCoreInspector(config)
        self.pose = YOLOPoseInspector(config)
        self.engine = DecisionEngine(config)
        self.repo = InspectionRepository()
        self.reports = DefectReportGenerator()

    def inspect(self, request: InspectRequest) -> InspectionResult:
        detection_result: ModuleResult

        # ---------------------------------------------------------
        # 1. Acquire Camera 1 and Camera 2 frames
        # ---------------------------------------------------------
        if request.camera1_video_path and request.camera2_video_path:

            first = self.frame_selector.select(
                request.camera1_video_path
            )

            second = self.frame_selector.select(
                request.camera2_video_path
            )

            camera1 = first.frame
            camera2 = second.frame

            detection_result = self._combine_detection_results(
                first.detection_result,
                second.detection_result,
            )

            camera1_detection_result = first.detection_result

        else:

            camera1, camera2 = self.capture.capture_pair(
                request.camera1_path,
                request.camera2_path,
            )

            camera1_detection_result, _ = self.detector.detect(
                camera1
            )

            camera2_detection_result, _ = self.detector.detect(
                camera2
            )

            detection_result = self._combine_detection_results(
                camera1_detection_result,
                camera2_detection_result,
            )

        # ---------------------------------------------------------
        # 2. Get Camera 1 YOLO bounding box
        # ---------------------------------------------------------
        detection_bbox = self._get_detection_bbox(
            camera1_detection_result
        )

        # If Camera 1 detection failed, PatchCore cannot localize
        # the three spring regions.
        if detection_bbox is None:

            module_results = [
                detection_result,
            ]

            decision, failures = self.engine.decide(
                module_results
            )

            result = InspectionResult(
                inspection_id=uuid4().hex,
                timestamp=datetime.now(timezone.utc),
                spring_id=request.spring_id,
                batch_id=request.batch_id,
                decision=decision,
                failures=failures,
                module_results=module_results,
            )

            result.report_path = self.reports.generate(
                result
            )

            self.repo.save(
                result,
                json.dumps(
                    result.model_dump(mode="json")
                ),
            )

            return result

        # ---------------------------------------------------------
        # 3. Run PatchCore + Pose modules
        #
        # IMPORTANT:
        # PatchCore receives the ORIGINAL Camera 1 image.
        # It then uses the YOLO bounding box to create:
        #   upper / shaft / lower
        # regional crops.
        # ---------------------------------------------------------
        with ThreadPoolExecutor(
            max_workers=self.config["inspection"][
                "max_parallel_workers"
            ]
        ) as pool:

            one = pool.submit(
                self.patchcore.inspect,
                camera1,
                detection_bbox,
            )

            two = pool.submit(
                self.pose.inspect_front,
                camera1,
            )

            three = pool.submit(
                self.pose.inspect_top,
                camera2,
            )

            module_results = [
                detection_result,
                one.result(),
                two.result(),
                three.result(),
            ]

        # ---------------------------------------------------------
        # 4. Save inspection artifacts
        # ---------------------------------------------------------
        inspection_id = uuid4().hex

        artifact_dir = (
            Path("reports")
            / inspection_id
        )

        artifact_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        camera1_path = (
            artifact_dir
            / "camera1.png"
        )

        camera2_path = (
            artifact_dir
            / "camera2.png"
        )

        cv2.imwrite(
            str(camera1_path),
            camera1,
        )

        cv2.imwrite(
            str(camera2_path),
            camera2,
        )

        # ---------------------------------------------------------
        # 5. Map module results by module ID
        # ---------------------------------------------------------
        module_by_id = {
            result.module: result
            for result in module_results
        }

        # Camera 1 source belongs to PatchCore module
        module_by_id[1].artifacts[
            "camera1_source"
        ] = str(camera1_path)

        # Camera 2 source belongs to Camera 2 Pose module
        module_by_id[3].artifacts[
            "camera2_source"
        ] = str(camera2_path)

        # ---------------------------------------------------------
        # 6. Save pose overlays
        # ---------------------------------------------------------
        self._save_pose_overlay(
            module_by_id[2],
            artifact_dir,
            "camera1_keypoints.png",
        )

        self._save_pose_overlay(
            module_by_id[3],
            artifact_dir,
            "camera2_hooks.png",
        )

        # ---------------------------------------------------------
        # 7. Save PatchCore anomaly heatmaps
        # ---------------------------------------------------------
        self._save_anomaly_heatmap(
            module_by_id[1],
            artifact_dir,
        )

        # ---------------------------------------------------------
        # 8. Final decision
        # ---------------------------------------------------------
        decision, failures = self.engine.decide(
            module_results
        )

        result = InspectionResult(
            inspection_id=inspection_id,
            timestamp=datetime.now(timezone.utc),
            spring_id=request.spring_id,
            batch_id=request.batch_id,
            decision=decision,
            failures=failures,
            module_results=module_results,
        )

        # ---------------------------------------------------------
        # 9. Generate report and save result
        # ---------------------------------------------------------
        result.report_path = self.reports.generate(
            result
        )

        self.repo.save(
            result,
            json.dumps(
                result.model_dump(mode="json")
            ),
        )

        return result

    @staticmethod
    def _get_detection_bbox(
        detection_result: ModuleResult,
    ) -> tuple[int, int, int, int] | None:
        """
        Extract the YOLO Camera 1 bounding box from the
        detector ModuleResult.

        YOLOSpringDetector stores the selected bbox in:
            result.details["bbox"]
        """

        if detection_result.state != ModuleState.PASS:
            return None

        bbox = detection_result.details.get(
            "bbox"
        )

        if bbox is None:
            return None

        if len(bbox) != 4:
            return None

        return tuple(
            int(value)
            for value in bbox
        )

    @staticmethod
    def _save_pose_overlay(
        result,
        folder: Path,
        filename: str,
    ) -> None:

        overlay = result.details.pop(
            "overlay",
            None,
        )

        if overlay is not None:

            path = folder / filename

            cv2.imwrite(
                str(path),
                np.asarray(
                    overlay,
                    dtype=np.uint8,
                ),
            )

            result.artifacts[
                "keypoint_overlay"
            ] = str(path)

    @staticmethod
    def _save_anomaly_heatmap(
        result,
        folder: Path,
    ) -> None:

        # Regional PatchCore stores anomaly maps
        # inside details["regions"].
        regions = result.details.get(
            "regions",
            {},
        )

        if not regions:
            return

        for region, region_data in regions.items():

            anomaly_map = region_data.get(
                "anomaly_map"
            )

            if anomaly_map is None:
                continue

            heatmap = np.asarray(
                anomaly_map,
                dtype=np.float32,
            )

            heatmap = cv2.normalize(
                heatmap,
                None,
                0,
                255,
                cv2.NORM_MINMAX,
            ).astype(np.uint8)

            path = (
                folder
                / f"anomaly_heatmap_{region}.png"
            )

            cv2.imwrite(
                str(path),
                cv2.applyColorMap(
                    heatmap,
                    cv2.COLORMAP_JET,
                ),
            )

            result.artifacts[
                f"anomaly_heatmap_{region}"
            ] = str(path)

    @staticmethod
    def _combine_detection_results(
        first: ModuleResult,
        second: ModuleResult,
    ) -> ModuleResult:

        if (
            first.state == ModuleState.SKIPPED
            and second.state == ModuleState.SKIPPED
        ):
            return ModuleResult(
                module=0,
                state=ModuleState.SKIPPED,
                message=(
                    "YOLO spring detector disabled."
                ),
            )

        if (
            first.state == ModuleState.PASS
            and second.state == ModuleState.PASS
        ):
            return ModuleResult(
                module=0,
                state=ModuleState.PASS,
                metrics={
                    "camera1_detection_confidence": (
                        first.metrics[
                            "detection_confidence"
                        ]
                    ),
                    "camera2_detection_confidence": (
                        second.metrics[
                            "detection_confidence"
                        ]
                    ),
                },
                details={
                    "camera1_bbox": first.details.get(
                        "bbox"
                    ),
                    "camera2_bbox": second.details.get(
                        "bbox"
                    ),
                },
            )

        messages = "; ".join(
            filter(
                None,
                [
                    first.message,
                    second.message,
                ],
            )
        )

        state = (
            ModuleState.ERROR
            if ModuleState.ERROR
            in (
                first.state,
                second.state,
            )
            else ModuleState.FAIL
        )

        return ModuleResult(
            module=0,
            state=state,
            message=(
                messages
                or "Spring was not confirmed in both camera feeds."
            ),
        )