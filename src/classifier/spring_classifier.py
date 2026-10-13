"""
Production spring classifier.

Pipeline:
    image path
        → YOLO11 pose (V6-2) → 7 keypoints
        → feature_extractor  → 26 features
        → Random Forest      → GOOD / BAD

Usage:
    from src.classifier.spring_classifier import SpringClassifier

    clf = SpringClassifier()
    result = clf.predict("path/to/image.jpg")
    print(result)
"""

import sys
import pathlib
import pickle
import numpy as np
from typing import Dict, Any

ROOT = pathlib.Path("D:/spring-defects")
sys.path.insert(0, str(ROOT / "src"))

from ultralytics import YOLO
from features.feature_extractor import extract_features

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH      = ROOT / "runs/pose/spring_pose_v6-2/weights/best.pt"
CLASSIFIER_PATH = ROOT / "runs/features/classifier.pkl"
KP_COLS         = [f"kp{i}_{c}" for i in range(7) for c in ("x", "y")]
GEO_COLS        = [
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


def _normalize_keypoints(kp_xy: np.ndarray) -> dict:
    """Translate to kpt0 origin, scale by kpt0→kpt6 distance."""
    kp       = kp_xy[:, :2].astype(float)
    origin   = kp[0]
    safe_len = max(float(np.linalg.norm(kp[6] - kp[0])), 1.0)
    normed   = (kp - origin) / safe_len
    return {f"kp{i}_{c}": float(normed[i, j])
            for i in range(7) for j, c in enumerate(("x", "y"))}


class SpringClassifier:
    """
    Load once, call predict() for each image.

    Parameters
    ----------
    model_path      : path to YOLO pose weights (.pt)
    classifier_path : path to trained classifier (.pkl)
    device          : YOLO inference device ("0", "cpu", etc.)
    """

    def __init__(
        self,
        model_path: pathlib.Path = MODEL_PATH,
        classifier_path: pathlib.Path = CLASSIFIER_PATH,
        device: str = "0",
    ):
        print(f"Loading YOLO model   : {model_path}")
        self.yolo   = YOLO(str(model_path))
        self.device = device

        print(f"Loading classifier   : {classifier_path}")
        with open(classifier_path, "rb") as f:
            ckpt = pickle.load(f)

        self.pipeline      = ckpt["pipeline"]
        self.model_name    = ckpt["model_name"]
        self.feature_cols  = ckpt["feature_cols"]
        print(f"Classifier loaded    : {self.model_name}\n")

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        Run full pipeline on one image.

        Parameters
        ----------
        image_path : str or Path to the image file

        Returns
        -------
        dict with keys:
            image         : filename
            verdict       : "GOOD" or "BAD"
            confidence    : probability of BAD class [0.0 – 1.0]
            keypoints     : list of 7 {x, y} dicts (absolute pixels)
            features      : dict of 13 geometric features
            error         : str or None (if detection failed)
        """
        image_path = pathlib.Path(image_path)

        result_template = {
            "image"     : image_path.name,
            "verdict"   : None,
            "confidence": None,
            "keypoints" : None,
            "features"  : None,
            "error"     : None,
        }

        # ── YOLO inference ────────────────────────────────────────────────────
        results = self.yolo(str(image_path), verbose=False, device=self.device)
        result  = results[0]

        if (result.keypoints is None
                or result.keypoints.xy is None
                or len(result.keypoints.xy) == 0):
            result_template["error"]   = "No spring detected"
            result_template["verdict"] = "UNKNOWN"
            return result_template

        # ── Pick best detection ───────────────────────────────────────────────
        if result.boxes is not None and len(result.boxes.conf) > 1:
            best_idx = int(result.boxes.conf.argmax())
        else:
            best_idx = 0

        kp_xy = result.keypoints.xy[best_idx].cpu().numpy()   # [7, 2]

        if result.keypoints.conf is not None:
            kp_vis = result.keypoints.conf[best_idx].cpu().numpy().reshape(-1, 1)
        else:
            kp_vis = np.ones((7, 1), dtype=float)

        keypoints = np.concatenate([kp_xy, kp_vis], axis=1)   # [7, 3]

        # ── Extract features ──────────────────────────────────────────────────
        img_h = result.orig_shape[0]

        try:
            geo = extract_features(keypoints)
        except Exception as e:
            result_template["error"]   = f"Feature extraction failed: {e}"
            result_template["verdict"] = "UNKNOWN"
            return result_template

        geo["total_length_ratio"] = geo["total_length_px"] / max(img_h, 1)
        kp_normed = _normalize_keypoints(keypoints)

        # ── Build feature vector ──────────────────────────────────────────────
        feature_dict = {**geo, **kp_normed}
        X = np.array([[feature_dict[col] for col in self.feature_cols]])

        # ── Classify ──────────────────────────────────────────────────────────
        label = int(self.pipeline.predict(X)[0])

        if hasattr(self.pipeline, "predict_proba"):
            proba      = self.pipeline.predict_proba(X)[0]
            confidence = float(proba[1])   # probability of BAD
        else:
            confidence = None

        if confidence is not None:
            if confidence >= 0.80:
                verdict = "BAD"
            elif confidence >= 0.50:
                verdict = "MANUAL_CHECK"
            else:
                verdict = "GOOD"
        else:
            verdict = "BAD" if label == 1 else "GOOD"

        return {
            "image"     : image_path.name,
            "verdict"   : verdict,
            "confidence": confidence,
            "keypoints" : [{"x": float(kp_xy[i, 0]),
                            "y": float(kp_xy[i, 1])} for i in range(7)],
            "features"  : {k: round(v, 4) for k, v in geo.items()},
            "error"     : None,
        }


# ── Quick test when run directly ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python spring_classifier.py <image_path>")
        sys.exit(1)

    clf    = SpringClassifier()
    result = clf.predict(sys.argv[1])

    print(f"\n{'=' * 50}")
    print(f"  Image   : {result['image']}")
    print(f"  Verdict : {result['verdict']}")
    if result['confidence'] is not None:
        print(f"  BAD probability : {result['confidence']:.3f}")
    if result['error']:
        print(f"  Error   : {result['error']}")
    print(f"{'=' * 50}")

    if result['features']:
        print("\n  Key features:")
        for k in ["total_length_ratio", "shaft_deviation_ratio",
                  "top_hook_angle_deg", "bottom_hook_angle_deg"]:
            print(f"    {k:<30} {result['features'].get(k, 'N/A')}")

    print(f"\n  Full result:\n{json.dumps(result, indent=2)}")