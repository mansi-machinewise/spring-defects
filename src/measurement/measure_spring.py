"""
Spring measurement engine.

Takes a single image, runs YOLO11-pose inference, extracts 7 keypoints,
calculates physical measurements in mm, and returns PASS/FAIL result.

Usage:
    python src/measurement/measure_spring.py --image <path_to_image>
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_PATH     = "D:/spring-defects/runs/pose/spring_pose_v6-2/weights/best.pt"
PIXELS_PER_MM  = 20.6978   # calibrated from training images

# Tolerances (mm)
TOTAL_LENGTH_MIN  = 119.50
TOTAL_LENGTH_MAX  = 121.00
HOOK_LENGTH_MIN   = 115.50
HOOK_LENGTH_MAX   = 118.50

# Keypoint names
KEYPOINT_NAMES = [
    "hook_tip_top",        # 0 - blue dot top
    "top_coil",            # 1 - red dot top
    "bottom_top_coil",     # 2
    "shaft_middle",        # 3
    "top_bottom_coil",     # 4
    "bottom_bottom_coil",  # 5 - red dot bottom
    "hook_tip_bottom",     # 6 - blue dot bottom
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def euclidean_mm(p1, p2, px_per_mm):
    dist_px = float(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))
    return dist_px, dist_px / px_per_mm


def pass_fail(value, min_val, max_val):
    return "PASS" if min_val <= value <= max_val else "FAIL"


# ── Main measurement function ─────────────────────────────────────────────────

def measure_spring(
    image_path: str,
    model_path: str = MODEL_PATH,
    pixels_per_mm: float = PIXELS_PER_MM,
    imgsz: int = 960,
    device: str = "0",
    conf: float = 0.25,
    save_visualization: bool = True,
    output_dir: str = "D:/spring-defects/runs/measurement",
):
    """
    Run full spring inspection on a single image.

    Returns a dict with keypoints, measurements, and PASS/FAIL.
    """

    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    model = YOLO(str(model_path))

    # ── Run inference ─────────────────────────────────────────────────────────
    results = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        device=device,
        verbose=False,
    )

    result = results[0]

    # ── Check detection ───────────────────────────────────────────────────────
    if result.keypoints is None or len(result.keypoints.data) == 0:
        return {
            "image": image_path.name,
            "status": "NO_DETECTION",
            "overall": "FAIL",
        }

    # Take highest confidence detection
    best_idx = int(np.argmax(result.boxes.conf.cpu().numpy()))
    kd = result.keypoints.data.cpu().numpy()[best_idx]  # (7, 3)
    box = result.boxes.xyxy.cpu().numpy()[best_idx].astype(int)

    # ── Extract keypoints ─────────────────────────────────────────────────────
    keypoints = {}
    for i, name in enumerate(KEYPOINT_NAMES):
        keypoints[name] = {
            "x": round(float(kd[i, 0]), 1),
            "y": round(float(kd[i, 1]), 1),
            "conf": round(float(kd[i, 2]), 4),
        }

    kpt0 = kd[0, :2]  # hook_tip_top
    kpt1 = kd[1, :2]  # top_coil
    kpt5 = kd[5, :2]  # bottom_bottom_coil
    kpt6 = kd[6, :2]  # hook_tip_bottom

    # ── Calculate measurements ────────────────────────────────────────────────
    total_px, total_mm = euclidean_mm(kpt0, kpt6, pixels_per_mm)
    hook_px,  hook_mm  = euclidean_mm(kpt1, kpt5, pixels_per_mm)

    total_result = pass_fail(total_mm, TOTAL_LENGTH_MIN, TOTAL_LENGTH_MAX)
    hook_result  = pass_fail(hook_mm,  HOOK_LENGTH_MIN,  HOOK_LENGTH_MAX)
    overall      = "PASS" if total_result == "PASS" and hook_result == "PASS" else "FAIL"

    # ── Build result dict ─────────────────────────────────────────────────────
    inspection = {
        "image": image_path.name,
        "status": "OK",
        "bounding_box": box.tolist(),
        "keypoints": keypoints,
        "measurements": {
            "total_length": {
                "pixels": round(total_px, 2),
                "mm": round(total_mm, 2),
                "min": TOTAL_LENGTH_MIN,
                "max": TOTAL_LENGTH_MAX,
                "result": total_result,
            },
            "hook_length": {
                "pixels": round(hook_px, 2),
                "mm": round(hook_mm, 2),
                "min": HOOK_LENGTH_MIN,
                "max": HOOK_LENGTH_MAX,
                "result": hook_result,
            },
        },
        "overall": overall,
    }

    # ── Print report ──────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"SPRING INSPECTION REPORT")
    print(f"{'='*50}")
    print(f"Image         : {image_path.name}")
    print(f"Detection     : {len(result.boxes)} spring(s) found")
    print(f"{'─'*50}")
    print(f"Total length  : {total_mm:.2f} mm  [{TOTAL_LENGTH_MIN}-{TOTAL_LENGTH_MAX}]  -> {total_result}")
    print(f"Hook length   : {hook_mm:.2f} mm  [{HOOK_LENGTH_MIN}-{HOOK_LENGTH_MAX}]  -> {hook_result}")
    print(f"{'─'*50}")
    print(f"OVERALL       : {overall}")
    print(f"{'='*50}\n")

    # ── Save JSON report ──────────────────────────────────────────────────────
    json_path = output_dir / (image_path.stem + "_result.json")
    with open(json_path, "w") as f:
        json.dump(inspection, f, indent=2)
    print(f"Report saved  : {json_path}")

    # ── Save visualization ────────────────────────────────────────────────────
    if save_visualization:
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]

        # bounding box
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 4)

        # keypoints
        for i, name in enumerate(KEYPOINT_NAMES):
            px, py = int(kd[i, 0]), int(kd[i, 1])
            color = (0, 0, 255) if i in [0, 6] else (0, 185, 185)
            cv2.circle(img, (px, py), 15, color, -1)
            cv2.putText(img, name[:6], (px + 10, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

        # measurement lines
        cv2.line(img,
                 (int(kpt0[0]), int(kpt0[1])),
                 (int(kpt6[0]), int(kpt6[1])),
                 (255, 0, 255), 3)
        cv2.line(img,
                 (int(kpt1[0]), int(kpt1[1])),
                 (int(kpt5[0]), int(kpt5[1])),
                 (0, 165, 255), 3)

        # result overlay
        color = (0, 200, 0) if overall == "PASS" else (0, 0, 255)
        cv2.putText(img, f"OVERALL: {overall}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 6)
        cv2.putText(img, f"Total: {total_mm:.2f}mm ({total_result})", (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 4)
        cv2.putText(img, f"Hook:  {hook_mm:.2f}mm ({hook_result})", (30, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 4)

        scale = min(1.0, 1200 / max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        vis_path = output_dir / (image_path.stem + "_visual.jpg")
        cv2.imwrite(str(vis_path), img)
        print(f"Visual saved  : {vis_path}")

    return inspection


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spring inspection — measure and PASS/FAIL a single image."
    )
    parser.add_argument("--image",   required=True, help="Path to spring image")
    parser.add_argument("--model",   default=MODEL_PATH)
    parser.add_argument("--px_per_mm", type=float, default=PIXELS_PER_MM)
    parser.add_argument("--imgsz",   type=int, default=960)
    parser.add_argument("--device",  default="0")
    parser.add_argument("--output",  default="D:/spring-defects/runs/measurement")
    parser.add_argument("--no_vis",  action="store_true")
    args = parser.parse_args()

    measure_spring(
        image_path=args.image,
        model_path=args.model,
        pixels_per_mm=args.px_per_mm,
        imgsz=args.imgsz,
        device=args.device,
        save_visualization=not args.no_vis,
        output_dir=args.output,
    )