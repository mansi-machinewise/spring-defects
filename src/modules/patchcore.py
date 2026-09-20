from __future__ import annotations

from pathlib import Path
import numpy as np
from src.models.schemas import ModuleResult, ModuleState


class PatchCoreInspector:
    """Anomalib PatchCore runtime boundary; only GOOD Camera 1 samples train it."""
    def __init__(self, config: dict) -> None:
        self.config = config
        self._inferencer = None

    def inspect(self, image: np.ndarray) -> ModuleResult:
        settings = self.config["patchcore"]
        if not settings["enabled"]:
            return ModuleResult(module=1, state=ModuleState.SKIPPED, message="PatchCore disabled; train on good springs then configure checkpoint.")
        checkpoint = Path(settings["checkpoint_path"])
        if not checkpoint.exists():
            return ModuleResult(module=1, state=ModuleState.ERROR, message=f"PatchCore checkpoint missing: {checkpoint}")
        try:
            # anomalib's public inferencer returns pred_score and anomaly_map.
            from anomalib.deploy import TorchInferencer
            if self._inferencer is None:
                self._inferencer = TorchInferencer(path=checkpoint, device="auto")
            prediction = self._inferencer.predict(image=(image * 255).astype(np.uint8))
            score = float(prediction.pred_score)
            failed = score > float(settings["anomaly_threshold"])
            anomaly_map = getattr(prediction, "anomaly_map", None)
            return ModuleResult(module=1, state=ModuleState.FAIL if failed else ModuleState.PASS,
                metrics={"anomaly_score": score}, details={"threshold": settings["anomaly_threshold"],
                "anomaly_map": anomaly_map.tolist() if anomaly_map is not None else None})
        except Exception as exc:
            return ModuleResult(module=1, state=ModuleState.ERROR, message=f"PatchCore inference error: {exc}")
