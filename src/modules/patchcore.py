from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.models.schemas import ModuleResult, ModuleState


class PatchCoreInspector:
    """
    Production Camera 1 regional PatchCore inspector.

    Preprocessing used during training and inference:

        Camera 1 image
            -> YOLO segmentation finds spring extent
            -> expand bbox
                 X padding = 30%
                 Y padding = 20%
            -> crop ORIGINAL image
            -> regional slicing
                 upper = 0.00 - 0.32
                 shaft = 0.24 - 0.76
                 lower = 0.68 - 1.00
            -> resize with aspect ratio
            -> center on 256x256 WHITE canvas
            -> regional PatchCore
            -> anomaly score + anomaly map

    PatchCore inference uses Anomalib 2.6.2 Engine.predict().
    """

    REGIONS = ("upper", "shaft", "lower")

    IMAGE_SIZE = 256

    VERTICAL_PADDING = 0.20
    HORIZONTAL_PADDING = 0.30

    REGION_RANGES = {
        "upper": (0.00, 0.32),
        "shaft": (0.24, 0.76),
        "lower": (0.68, 1.00),
    }

    def __init__(self, config: dict) -> None:

        # ---------------------------------------------------------
        # Accept either:
        #
        # 1. Full application config:
        #    config["patchcore"]
        #
        # OR
        #
        # 2. PatchCore-only config:
        #    config
        # ---------------------------------------------------------
        if "patchcore" in config:
            self.config = config
            patchcore_config = config.get("patchcore", {})
        else:
            patchcore_config = config
            self.config = {"patchcore": patchcore_config}

        self._engines: dict[str, Any] = {}
        self._models: dict[str, Any] = {}

        # ---------------------------------------------------------
        # PatchCore checkpoints
        # ---------------------------------------------------------
        self.checkpoint_root = Path(
            patchcore_config.get(
                "checkpoint_root",
                "models/patchcore_aligned",
            )
        )

        # ---------------------------------------------------------
        # Regional thresholds
        # ---------------------------------------------------------
        self.thresholds = patchcore_config.get(
            "regional_thresholds",
            {
                "upper": 0.5,
                "shaft": 0.5,
                "lower": 0.5,
            },
        )

        self.default_threshold = float(
            patchcore_config.get(
                "anomaly_threshold",
                0.5,
            )
        )

        # ---------------------------------------------------------
        # Segmentation configuration
        # ---------------------------------------------------------
        self.segmentation_enabled = bool(
            patchcore_config.get(
                "segmentation_enabled",
                True,
            )
        )

        self.segmentation_checkpoint = Path(
            patchcore_config.get(
                "segmentation_checkpoint",
                "runs/segment/runs/segment/spring_seg_v2/weights/best.pt",
            )
        )

        # IMPORTANT:
        # Training dataset was generated using confidence 0.10.
        self.segmentation_confidence = float(
            patchcore_config.get(
                "segmentation_confidence",
                0.10,
            )
        )

        self.segmentation_image_size = int(
            patchcore_config.get(
                "segmentation_image_size",
                960,
            )
        )

        self.segmentation_model = None

    # =========================================================
    # SEGMENTATION MODEL
    # =========================================================

    def _load_segmentation_model(self):

        if self.segmentation_model is not None:
            return self.segmentation_model

        if not self.segmentation_checkpoint.exists():
            raise FileNotFoundError(
                "Segmentation checkpoint not found: "
                f"{self.segmentation_checkpoint}"
            )

        from ultralytics import YOLO

        self.segmentation_model = YOLO(
            str(self.segmentation_checkpoint)
        )

        return self.segmentation_model

    # =========================================================
    # FIND SPRING EXTENT USING SEGMENTATION
    # =========================================================

    def _get_segmentation_extent(
        self,
        image: np.ndarray,
    ) -> tuple[int, int, int, int] | None:

        segmenter = self._load_segmentation_model()

        results = segmenter.predict(
            image,
            imgsz=self.segmentation_image_size,
            conf=self.segmentation_confidence,
            device=0,
            verbose=False,
        )

        if not results:
            return None

        result = results[0]

        if result.masks is None:
            return None

        masks = (
            result.masks.data
            .detach()
            .cpu()
            .numpy()
        )

        if len(masks) == 0:
            return None

        # Combine all detected spring masks.
        mask = np.any(
            masks > 0.5,
            axis=0,
        ).astype(np.uint8)

        # Resize mask back to original image dimensions.
        mask = cv2.resize(
            mask,
            (
                image.shape[1],
                image.shape[0],
            ),
            interpolation=cv2.INTER_NEAREST,
        )

        ys, xs = np.where(mask > 0)

        if len(xs) == 0 or len(ys) == 0:
            return None

        x1 = int(xs.min())
        x2 = int(xs.max()) + 1

        y1 = int(ys.min())
        y2 = int(ys.max()) + 1

        return x1, y1, x2, y2

    # =========================================================
    # EXPAND SPRING CROP
    # =========================================================

    @classmethod
    def _expand_crop(
        cls,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:

        x1, y1, x2, y2 = bbox

        height, width = image.shape[:2]

        spring_width = x2 - x1
        spring_height = y2 - y1

        pad_x = int(
            spring_width * cls.HORIZONTAL_PADDING
        )

        pad_y = int(
            spring_height * cls.VERTICAL_PADDING
        )

        crop_x1 = max(
            0,
            x1 - pad_x,
        )

        crop_x2 = min(
            width,
            x2 + pad_x,
        )

        crop_y1 = max(
            0,
            y1 - pad_y,
        )

        crop_y2 = min(
            height,
            y2 + pad_y,
        )

        return (
            crop_x1,
            crop_y1,
            crop_x2,
            crop_y2,
        )

    # =========================================================
    # RESIZE WITH WHITE PADDING
    # =========================================================

    @classmethod
    def _resize_with_padding(
        cls,
        image: np.ndarray,
        size: int = IMAGE_SIZE,
    ) -> np.ndarray | None:

        h, w = image.shape[:2]

        if h == 0 or w == 0:
            return None

        scale = min(
            size / w,
            size / h,
        )

        new_w = max(
            1,
            int(round(w * scale)),
        )

        new_h = max(
            1,
            int(round(h * scale)),
        )

        resized = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA,
        )

        # Exact same white canvas used during training.
        canvas = np.full(
            (size, size, 3),
            255,
            dtype=np.uint8,
        )

        x = (size - new_w) // 2
        y = (size - new_h) // 2

        canvas[
            y:y + new_h,
            x:x + new_w,
        ] = resized

        return canvas

    # =========================================================
    # CREATE PATCHCORE REGIONS
    # =========================================================

    @classmethod
    def _crop_regions(
        cls,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> dict[str, np.ndarray]:

        x1, y1, x2, y2 = cls._expand_crop(
            image,
            bbox,
        )

        spring_crop = image[
            y1:y2,
            x1:x2,
        ]

        if spring_crop.size == 0:
            raise ValueError(
                "Expanded spring crop is empty."
            )

        spring_height = spring_crop.shape[0]

        crops: dict[str, np.ndarray] = {}

        for region_name, (
            start_ratio,
            end_ratio,
        ) in cls.REGION_RANGES.items():

            start_y = int(
                spring_height * start_ratio
            )

            end_y = int(
                spring_height * end_ratio
            )

            if end_y <= start_y:
                raise ValueError(
                    f"Invalid {region_name} region."
                )

            region_crop = spring_crop[
                start_y:end_y,
                :,
            ]

            if region_crop.size == 0:
                raise ValueError(
                    f"{region_name.upper()} PatchCore crop is empty."
                )

            region_crop = cls._resize_with_padding(
                region_crop,
                cls.IMAGE_SIZE,
            )

            if region_crop is None:
                raise ValueError(
                    f"Failed to resize {region_name} crop."
                )

            crops[region_name] = region_crop

        return crops

    # =========================================================
    # PATCHCORE CHECKPOINT
    # =========================================================

    def _checkpoint_path(
        self,
        region: str,
    ) -> Path:

        return (
            self.checkpoint_root
            / region
            / "Patchcore"
            / f"spring_{region}_aligned"
            / "v0"
            / "weights"
            / "lightning"
            / "model.ckpt"
        )

    # =========================================================
    # LOAD PATCHCORE MODEL
    # =========================================================

    def _load_patchcore(
        self,
        region: str,
    ):

        if region in self._models:
            return (
                self._models[region],
                self._engines[region],
            )

        checkpoint = self._checkpoint_path(
            region,
        )

        if not checkpoint.exists():
            raise FileNotFoundError(
                f"{region.upper()} PatchCore checkpoint "
                f"not found: {checkpoint}"
            )

        from anomalib.engine import Engine
        from anomalib.models import Patchcore

        model = Patchcore(
            backbone="wide_resnet50_2",
        )

        engine = Engine()

        self._models[region] = model
        self._engines[region] = engine

        return model, engine

    # =========================================================
    # SINGLE REGION INFERENCE
    # =========================================================

    def _predict_region(
        self,
        region: str,
        crop: np.ndarray,
    ) -> tuple[float, np.ndarray | None]:

        model, engine = self._load_patchcore(region)

        # ---------------------------------------------------------
        # Anomalib 2.6.2 Engine.predict() expects a data_path.
        # Create a temporary directory containing this single crop.
        # ---------------------------------------------------------
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:

            temp_path = Path(temp_dir) / f"{region}.png"

            success = cv2.imwrite(
                str(temp_path),
                crop,
            )

            if not success:
                raise RuntimeError(
                    f"Failed to write temporary PatchCore "
                    f"crop for {region.upper()}."
                )

            predictions = engine.predict(
                model=model,
                ckpt_path=str(
                    self._checkpoint_path(region)
                ),
                data_path=temp_dir,
                return_predictions=True,
            )

        if not predictions:
            raise RuntimeError(
                f"No PatchCore prediction returned "
                f"for {region.upper()}."
            )

        prediction = predictions[0]

        pred_score = getattr(
            prediction,
            "pred_score",
            None,
        )

        if pred_score is None:
            raise RuntimeError(
                f"PatchCore prediction for "
                f"{region.upper()} has no pred_score."
            )

        if hasattr(pred_score, "detach"):
            score = float(
                pred_score
                .detach()
                .cpu()
                .reshape(-1)[0]
                .item()
            )
        else:
            score = float(
                np.asarray(pred_score)
                .reshape(-1)[0]
            )

        anomaly_map = getattr(
            prediction,
            "anomaly_map",
            None,
        )

        if anomaly_map is not None:

            if hasattr(anomaly_map, "detach"):
                anomaly_map = (
                    anomaly_map
                    .detach()
                    .cpu()
                    .numpy()
                )
            else:
                anomaly_map = np.asarray(
                    anomaly_map
                )

            anomaly_map = np.squeeze(
                anomaly_map
            )

        return score, anomaly_map

    # =========================================================
    # PUBLIC INSPECTION
    # =========================================================

    def inspect(
        self,
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> ModuleResult:

        settings = self.config.get(
            "patchcore",
            {},
        )

        if not settings.get(
            "enabled",
            True,
        ):
            return ModuleResult(
                module=1,
                state=ModuleState.SKIPPED,
                message="PatchCore disabled.",
            )

        if image is None or image.size == 0:
            return ModuleResult(
                module=1,
                state=ModuleState.ERROR,
                message="PatchCore received an empty image.",
            )

        try:

            # -------------------------------------------------
            # Determine spring extent.
            #
            # Training used segmentation-derived extent.
            # Therefore segmentation remains the primary source.
            #
            # If segmentation is disabled or unavailable,
            # fall back to the YOLO detector bbox.
            # -------------------------------------------------

            segmentation_bbox = None

            if self.segmentation_enabled:

                segmentation_bbox = (
                    self._get_segmentation_extent(
                        image
                    )
                )

            if segmentation_bbox is not None:

                inspection_bbox = segmentation_bbox
                bbox_source = "segmentation"

            else:

                if bbox is None:
                    return ModuleResult(
                        module=1,
                        state=ModuleState.ERROR,
                        message=(
                            "PatchCore could not determine "
                            "spring extent from segmentation "
                            "and no detector bbox was provided."
                        ),
                    )

                inspection_bbox = tuple(
                    int(v)
                    for v in bbox
                )

                bbox_source = "yolo_detector_fallback"

            # -------------------------------------------------
            # Create exactly the same regional crops used
            # during PatchCore training.
            # -------------------------------------------------

            crops = self._crop_regions(
                image,
                inspection_bbox,
            )

            results = {}
            failed_regions = []

            # -------------------------------------------------
            # Run three regional PatchCore models.
            # -------------------------------------------------

            for region in self.REGIONS:

                crop = crops[region]

                score, anomaly_map = (
                    self._predict_region(
                        region,
                        crop,
                    )
                )

                threshold = float(
                    self.thresholds.get(
                        region,
                        self.default_threshold,
                    )
                )

                failed = score > threshold

                if failed:
                    failed_regions.append(
                        region
                    )

                results[region] = {
                    "anomaly_score": round(
                        score,
                        4,
                    ),
                    "threshold": threshold,
                    "failed": failed,
                    "anomaly_map": (
                        anomaly_map.tolist()
                        if anomaly_map is not None
                        else None
                    ),
                }

            overall_failed = (
                len(failed_regions) > 0
            )

            return ModuleResult(
                module=1,
                state=(
                    ModuleState.FAIL
                    if overall_failed
                    else ModuleState.PASS
                ),
                metrics={
                    "upper_anomaly_score": results[
                        "upper"
                    ]["anomaly_score"],
                    "shaft_anomaly_score": results[
                        "shaft"
                    ]["anomaly_score"],
                    "lower_anomaly_score": results[
                        "lower"
                    ]["anomaly_score"],
                },
                details={
                    "segmentation_bbox": (
                        segmentation_bbox
                    ),
                    "inspection_bbox": (
                        inspection_bbox
                    ),
                    "bbox_source": bbox_source,
                    "regions": results,
                    "failed_regions": failed_regions,
                },
            )

        except Exception as exc:

            return ModuleResult(
                module=1,
                state=ModuleState.ERROR,
                message=(
                    "Regional PatchCore inference error: "
                    f"{exc}"
                ),
            )