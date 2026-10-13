"""
Run best.pt on all 500 training images and save geometric features to CSV.

Labels:
    001-100  →  0  (GOOD)
    101-200  →  1  (BAD)
    201-350  →  0  (GOOD)
    351-500  →  1  (BAD)

Total: 250 GOOD, 250 BAD

Output:
    D:/spring-defects/runs/features/features.csv

Usage:
    D:\\spring-defects\\.venv\\Scripts\\python.exe D:\\spring-defects\\scripts\\extract_features.py
"""

import sys
import csv
import pathlib
import numpy as np

# ── Allow importing from src/ ─────────────────────────────────────────────────
ROOT = pathlib.Path("D:/spring-defects")
sys.path.insert(0, str(ROOT / "src"))

from ultralytics import YOLO
from features.feature_extractor import extract_features

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = ROOT / "runs/pose/spring_pose_v6-2/weights/best.pt"
IMAGES_DIR = ROOT / "data/yolo_pose/train/images"
OUTPUT_DIR = ROOT / "runs/features"
OUTPUT_CSV = OUTPUT_DIR / "features.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Label assignment ──────────────────────────────────────────────────────────
def get_label(i: int) -> int:
    """
    Return 0 (GOOD) or 1 (BAD) for spring number i.
    001-100  GOOD
    101-200  BAD
    201-350  GOOD
    351-500  BAD
    """
    if 1 <= i <= 100:
        return 0
    elif 101 <= i <= 200:
        return 1
    elif 201 <= i <= 350:
        return 0
    elif 351 <= i <= 500:
        return 1
    else:
        raise ValueError(f"Spring number {i} out of expected range 1-500")

# ── Feature columns ───────────────────────────────────────────────────────────
FEATURE_COLS = [
    "total_length_px",
    "total_length_ratio",
    "tilt_angle_deg",
    "shaft_deviation_px",
    "shaft_deviation_ratio",
    "top_hook_offset_px",
    "top_hook_offset_ratio",
    "top_hook_angle_deg",
    "bottom_hook_offset_px",
    "bottom_hook_offset_ratio",
    "bottom_hook_angle_deg",
    "top_coil_gap_ratio",
    "bottom_coil_gap_ratio",
]

KP_COLS = [f"kp{i}_{c}" for i in range(7) for c in ("x", "y")]
# produces: kp0_x, kp0_y, kp1_x, kp1_y, ... kp6_x, kp6_y

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading model: {MODEL_PATH}\n")
model = YOLO(str(MODEL_PATH))

# ── Process all 500 images ────────────────────────────────────────────────────
rows   = []
failed = []

def normalize_keypoints(kp_xy: np.ndarray) -> dict:
    """Translate to kpt0 origin, scale by kpt0→kpt6 distance."""
    kp        = kp_xy[:, :2].astype(float)
    origin    = kp[0]
    safe_len  = max(float(np.linalg.norm(kp[6] - kp[0])), 1.0)
    normed    = (kp - origin) / safe_len
    return {f"kp{i}_{c}": round(float(normed[i, j]), 6)
            for i in range(7) for j, c in enumerate(("x", "y"))}

for i in range(1, 501):
    ext      = ".jpeg" if i <= 200 else ".jpg"
    img_name = f"spring_{i:03d}{ext}"
    img_path = IMAGES_DIR / img_name
    if not img_path.exists():
        img_path = ROOT / "data/yolo_pose/extra/images" / img_name

    import cv2
    label    = get_label(i)
    tag      = "GOOD" if label == 0 else "BAD "

    # ── Check file exists ─────────────────────────────────────────────────────
    if not img_path.exists():
        print(f"  [MISSING]      {img_name}")
        failed.append(img_name)
        continue

    # ── Run inference ─────────────────────────────────────────────────────────
    results = model(str(img_path), verbose=False)
    result  = results[0]
    img_h = result.orig_shape[0]

    if (result.keypoints is None
            or result.keypoints.xy is None
            or len(result.keypoints.xy) == 0):
        print(f"  [NO DETECTION] {img_name}")
        failed.append(img_name)
        continue

    # ── Pick best detection (highest box confidence) ──────────────────────────
    if result.boxes is not None and len(result.boxes.conf) > 1:
        best_idx = int(result.boxes.conf.argmax())
    else:
        best_idx = 0

    kp_xy = result.keypoints.xy[best_idx].cpu().numpy()       # [7, 2]

    if result.keypoints.conf is not None:
        kp_vis = result.keypoints.conf[best_idx].cpu().numpy().reshape(-1, 1)
    else:
        kp_vis = np.ones((7, 1), dtype=float)

    keypoints = np.concatenate([kp_xy, kp_vis], axis=1)       # [7, 3]

    # ── Extract features ──────────────────────────────────────────────────────
    try:
        features = extract_features(keypoints)
        features["total_length_ratio"] = round(features["total_length_px"] / img_h, 6)
        kp_normed = normalize_keypoints(keypoints)
    except Exception as e:
        print(f"  [FEATURE ERR]  {img_name} → {e}")
        failed.append(img_name)
        continue

    # ── Accumulate row ────────────────────────────────────────────────────────
    row = {"image": img_name, "label": label}
    row.update({k: round(features[k], 4) for k in FEATURE_COLS})
    row.update(kp_normed)
    rows.append(row)

    print(
        f"  [{tag}] {img_name}"
        f"  tilt={features['tilt_angle_deg']:5.2f}°"
        f"  shaft={features['shaft_deviation_px']:6.1f}px"
        f"  top_hook={features['top_hook_angle_deg']:5.2f}°"
        f"  bot_hook={features['bottom_hook_angle_deg']:5.2f}°"
    )

# ── Write CSV ─────────────────────────────────────────────────────────────────
fieldnames = ["image", "label"] + FEATURE_COLS + KP_COLS

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# ── Summary ───────────────────────────────────────────────────────────────────
good_rows = [r for r in rows if r["label"] == 0]
bad_rows  = [r for r in rows if r["label"] == 1]

print(f"\n{'=' * 60}")
print(f"Processed : {len(rows)} images  ({len(good_rows)} GOOD, {len(bad_rows)} BAD)")
if failed:
    print(f"Skipped   : {len(failed)} → {failed}")
print(f"CSV saved : {OUTPUT_CSV}")
print(f"{'=' * 60}")

# ── Per-feature mean comparison ───────────────────────────────────────────────
if good_rows and bad_rows:
    print(f"\n{'Feature':<30} {'GOOD mean':>12} {'BAD mean':>12} {'Diff':>10}")
    print("-" * 66)
    for col in FEATURE_COLS:
        g_mean = np.mean([r[col] for r in good_rows])
        b_mean = np.mean([r[col] for r in bad_rows])
        diff   = b_mean - g_mean
        print(f"  {col:<28} {g_mean:>12.4f} {b_mean:>12.4f} {diff:>+10.4f}")