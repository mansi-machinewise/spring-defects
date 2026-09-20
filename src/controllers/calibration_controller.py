from fastapi import APIRouter

from src.models.schemas import CalibrationRequest, CalibrationResult


def build_router(service) -> APIRouter:
    router = APIRouter(prefix="/api/v1/calibration", tags=["calibration"])

    @router.post("/scale", response_model=CalibrationResult)
    def calculate_scale(request: CalibrationRequest) -> CalibrationResult:
        """Compute px/mm from a measured reference; set apply=true to save it."""
        return service.calculate(request)

    return router
