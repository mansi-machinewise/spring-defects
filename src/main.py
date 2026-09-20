from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from src.controllers.calibration_controller import build_router as build_calibration_router
from src.controllers.inspection_controller import build_router
from src.services.config import load_config
from src.services.calibration_service import CalibrationService
from src.services.inspection_service import InspectionService

config = load_config()
inspection_service = InspectionService(config)
calibration_service = CalibrationService(config)
app = FastAPI(title="Spring Defect Detection API", version="0.1.0")
app.include_router(build_router(inspection_service))
app.include_router(build_calibration_router(calibration_service))
app.mount("/reports", StaticFiles(directory="reports"), name="reports")


@app.get("/health")
def health():
    return {"status": "ok", "camera_mode": config["camera"]["mode"]}
