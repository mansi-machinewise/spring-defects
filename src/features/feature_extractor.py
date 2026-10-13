"""
Geometric feature extractor for spring keypoints.

Input  : 7 keypoints [7, 3] (x, y, visibility) in pixel coordinates
Output : dict of geometric features

Keypoint index reference:
    0 = hook_tip_top
    1 = top_coil
    2 = bottom_top_coil
    3 = shaft_middle
    4 = top_bottom_coil
    5 = bottom_bottom_coil
    6 = hook_tip_bottom

Note: uses manual 2D cross product for NumPy 2.x compatibility.
"""

import numpy as np
from typing import Dict


# ── Keypoint indices ──────────────────────────────────────────────────────────
KPT_HOOK_TIP_TOP = 0
KPT_TOP_COIL     = 1
KPT_BOT_TOP_COIL = 2
KPT_SHAFT_MIDDLE = 3
KPT_TOP_BOT_COIL = 4
KPT_BOT_BOT_COIL = 5
KPT_HOOK_TIP_BOT = 6


# ── Internal helpers ──────────────────────────────────────────────────────────

def _dist(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 2-D points."""
    return float(np.linalg.norm(b - a))


def _cross2d(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    2D cross product (scalar) of two 2-D vectors.
    v1 x v2 = v1[0]*v2[1] - v1[1]*v2[0]
    Compatible with NumPy 2.x which dropped 2D vector support in np.cross.
    """
    return float(v1[0] * v2[1] - v1[1] * v2[0])


def _point_to_line_dist(point: np.ndarray,
                        line_a: np.ndarray,
                        line_b: np.ndarray) -> float:
    """
    Perpendicular distance from 'point' to the infinite line through
    line_a and line_b.
    Falls back to distance from line_a when the two line points coincide.
    """
    d    = line_b - line_a
    norm = float(np.linalg.norm(d))
    if norm < 1e-6:
        return _dist(point, line_a)
    return abs(_cross2d(d, line_a - point)) / norm


def _angle_from_vertical_deg(p_from: np.ndarray, p_to: np.ndarray) -> float:
    """
    Angle (degrees) between the vector p_from->p_to and the vertical axis.
    Image coordinates: y increases downward, so vertical is (0, +1).
    Returns a value in [0, 90].
    """
    dx = float(p_to[0] - p_from[0])
    dy = float(p_to[1] - p_from[1])
    return float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-9)))


# ── Public API ────────────────────────────────────────────────────────────────

def extract_features(keypoints: np.ndarray) -> Dict[str, float]:
    """
    Extract geometric features from the 7 spring keypoints.

    Parameters
    ----------
    keypoints : np.ndarray, shape [7, 3] or [7, 2]
        Each row is (x, y) or (x, y, visibility) in pixel coordinates.
        Coordinates must be in absolute pixels (not normalised).

    Returns
    -------
    dict[str, float] with the following keys:

    Overall geometry
        total_length_px          pixel distance  kpt0 -> kpt6
        tilt_angle_deg           angle of kpt0->kpt6 from vertical (deg)

    Shaft straightness
        shaft_deviation_px       perpendicular distance of kpt3 from line kpt2->kpt4
        shaft_deviation_ratio    shaft_deviation_px / total_length_px

    Top hook
        top_hook_offset_px       horizontal |x_kpt0 - x_kpt1|
        top_hook_offset_ratio    top_hook_offset_px / total_length_px
        top_hook_angle_deg       angle of kpt0->kpt1 from vertical (deg)

    Bottom hook
        bottom_hook_offset_px    horizontal |x_kpt6 - x_kpt5|
        bottom_hook_offset_ratio bottom_hook_offset_px / total_length_px
        bottom_hook_angle_deg    angle of kpt6->kpt5 from vertical (deg)

    Coil gaps
        top_coil_gap_ratio       |y_kpt2 - y_kpt1| / total_length_px
        bottom_coil_gap_ratio    |y_kpt5 - y_kpt4| / total_length_px
    """
    if keypoints.shape[0] != 7 or keypoints.shape[1] < 2:
        raise ValueError(
            f"Expected keypoints shape (7, 2) or (7, 3), got {keypoints.shape}"
        )

    kp = keypoints[:, :2].astype(float)

    p0 = kp[KPT_HOOK_TIP_TOP]
    p1 = kp[KPT_TOP_COIL]
    p2 = kp[KPT_BOT_TOP_COIL]
    p3 = kp[KPT_SHAFT_MIDDLE]
    p4 = kp[KPT_TOP_BOT_COIL]
    p5 = kp[KPT_BOT_BOT_COIL]
    p6 = kp[KPT_HOOK_TIP_BOT]

    # Overall geometry
    total_length_px = _dist(p0, p6)
    safe_total      = max(total_length_px, 1.0)
    tilt_angle_deg  = _angle_from_vertical_deg(p0, p6)

    # Shaft straightness
    shaft_deviation_px    = _point_to_line_dist(p3, p2, p4)
    shaft_deviation_ratio = shaft_deviation_px / safe_total

    # Top hook
    top_hook_offset_px    = abs(p0[0] - p1[0])
    top_hook_offset_ratio = top_hook_offset_px / safe_total
    top_hook_angle_deg    = _angle_from_vertical_deg(p0, p1)

    # Bottom hook
    bottom_hook_offset_px    = abs(p6[0] - p5[0])
    bottom_hook_offset_ratio = bottom_hook_offset_px / safe_total
    bottom_hook_angle_deg    = _angle_from_vertical_deg(p6, p5)

    # Coil gaps
    top_coil_gap_ratio    = abs(p2[1] - p1[1]) / safe_total
    bottom_coil_gap_ratio = abs(p5[1] - p4[1]) / safe_total

    return {
        "total_length_px"         : total_length_px,
        "tilt_angle_deg"          : tilt_angle_deg,
        "shaft_deviation_px"      : shaft_deviation_px,
        "shaft_deviation_ratio"   : shaft_deviation_ratio,
        "top_hook_offset_px"      : top_hook_offset_px,
        "top_hook_offset_ratio"   : top_hook_offset_ratio,
        "top_hook_angle_deg"      : top_hook_angle_deg,
        "bottom_hook_offset_px"   : bottom_hook_offset_px,
        "bottom_hook_offset_ratio": bottom_hook_offset_ratio,
        "bottom_hook_angle_deg"   : bottom_hook_angle_deg,
        "top_coil_gap_ratio"      : top_coil_gap_ratio,
        "bottom_coil_gap_ratio"   : bottom_coil_gap_ratio,
    }