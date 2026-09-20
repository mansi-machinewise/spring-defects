from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from pydantic import BaseModel, Field


class ModuleState(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


class ModuleResult(BaseModel):
    module: int
    state: ModuleState
    message: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)


class Failure(BaseModel):
    module: int
    reason: str
    value: float | None = None
    limits: dict[str, float] | None = None


class InspectionResult(BaseModel):
    inspection_id: str
    timestamp: datetime
    spring_id: str | None = None
    batch_id: str | None = None
    decision: ModuleState
    failures: list[Failure] = Field(default_factory=list)
    module_results: list[ModuleResult]
    report_path: str | None = None


class InspectRequest(BaseModel):
    spring_id: str | None = None
    batch_id: str | None = None
    camera1_path: str | None = None
    camera2_path: str | None = None
    camera1_video_path: str | None = None
    camera2_video_path: str | None = None


class CalibrationRequest(BaseModel):
    known_size_px: float = Field(gt=0, description="Measured pixel length of calibration reference")
    known_size_mm: float = Field(gt=0, description="Physical length of that reference in millimetres")
    apply: bool = Field(default=False, description="Persist the computed scale into config/tolerances.yaml")


class CalibrationResult(BaseModel):
    px_per_mm: float
    mm_per_px: float
    applied: bool
