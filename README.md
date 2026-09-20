# Spring defect detection

MVC-based visual inspection service for two-camera spring inspection. YOLO11-detect confirms presence and selects the sharpest spring frame in the 14–15 s window. Camera 1 then fans out to PatchCore and YOLO11-pose dimensional measurement; Camera 2 uses YOLO11-pose hook tips plus OpenCV coil-centre geometry. The decision service combines all outcomes.

## Quick start (simulation mode)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "."
uvicorn src.main:app --reload
```

Open `http://127.0.0.1:8000/docs`. Send `POST /api/v1/inspections/simulate` to exercise the full workflow. For the dashboard, in a second terminal run `streamlit run src/views/dashboard.py`.

## Production setup order

1. Calibrate Camera 1 using a known-size target. Measure its pixel length in an unscaled Camera 1 frame, then call `POST /api/v1/calibration/scale` with `known_size_px`, `known_size_mm`, and `apply: true`. This saves `calibration.px_per_mm` in `config/tolerances.yaml`.
2. Collect good Camera 1 images and train/export PatchCore; set `patchcore.enabled` and its model path.
3. Label `pt1`…`pt7` in CVAT, export YOLO pose format, train YOLO11-pose, then set the front/top checkpoint paths.
4. Train YOLO11-detect from spring bounding-box labels for frame confirmation and selection.
5. Tune the Camera 2 OpenCV coil-centre detection, hook-separation tolerance, and dimensional tolerances against caliper-validated parts.
5. Set `camera.mode` to `opencv`, `basler`, or `plc` and implement the appropriate driver adapter in `camera/capture.py`.

The service deliberately marks unavailable ML modules as `SKIPPED`, rather than letting an untrained placeholder accept or reject a physical part. Enable `inspection.require_all_modules` before production.
