from __future__ import annotations

from pathlib import Path
import yaml

from src.models.schemas import CalibrationRequest, CalibrationResult
from src.preprocessing.calibration import Calibration


class CalibrationService:
    """Owns calibration calculation and the optional configuration update."""
    def __init__(self, config: dict, config_path: str = "config/tolerances.yaml") -> None:
        self.config, self.config_path = config, Path(config_path)

    def calculate(self, request: CalibrationRequest) -> CalibrationResult:
        scale = Calibration.from_reference(request.known_size_px, request.known_size_mm)
        if request.apply:
            self.config["calibration"]["px_per_mm"] = scale.px_per_mm
            self.config_path.write_text(yaml.safe_dump(self.config, sort_keys=False), encoding="utf-8")
        return CalibrationResult(px_per_mm=scale.px_per_mm, mm_per_px=1 / scale.px_per_mm, applied=request.apply)
