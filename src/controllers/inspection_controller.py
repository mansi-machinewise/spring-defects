from fastapi import APIRouter, HTTPException
from src.models.schemas import InspectRequest, InspectionResult


def build_router(service) -> APIRouter:
    router = APIRouter(prefix="/api/v1/inspections", tags=["inspections"])

    @router.post("", response_model=InspectionResult)
    def inspect(request: InspectRequest) -> InspectionResult:
        try:
            return service.inspect(request)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.post("/simulate", response_model=InspectionResult)
    def simulate() -> InspectionResult:
        return service.inspect(InspectRequest(spring_id="SIMULATED"))

    @router.get("/latest")
    def latest(limit: int = 20):
        return [{"inspection_id": row.inspection_id, "timestamp": row.timestamp,
                 "spring_id": row.spring_id, "batch_id": row.batch_id,
                 "decision": row.decision, "report_path": row.report_path}
                for row in service.repo.latest(limit)]

    return router
