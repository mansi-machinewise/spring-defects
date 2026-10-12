"""
src/api/app.py
Run from project root:  python -m uvicorn src.api.app:app --reload --port 8000
Open:                   http://localhost:8000
"""
import os
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from src.modules.classifier import SpringClassifier

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = os.getenv("SPRING_WEIGHTS",
                    str(ROOT / "runs/classify/spring_cls_v1/weights/best.pt"))
UNCERTAIN_THRESHOLD = float(os.getenv("SPRING_UNCERTAIN", "0.0"))
MAX_BYTES = 15 * 1024 * 1024

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model ONCE at startup, not per request
    state["clf"] = SpringClassifier(WEIGHTS, uncertain_threshold=UNCERTAIN_THRESHOLD)
    yield
    state.clear()


app = FastAPI(title="Spring Inspection - Camera 1", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    clf = state.get("clf")
    return {"status": "ok" if clf else "loading",
            "weights": clf.weights if clf else None,
            "imgsz": clf.imgsz if clf else None,
            "classes": list(clf.names.values()) if clf else None}


@app.post("/api/camera1/inference")
async def camera1_inference(image: UploadFile = File(...)):
    data = await image.read()
    if not data:
        raise HTTPException(400, "Empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image too large (max 15 MB)")
    try:
        img = Image.open(BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Not a valid image")
    return state["clf"].predict(img)


# Frontend (must be mounted last so /api routes win)
app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
