from __future__ import annotations

import math
import numpy as np

KEYPOINT_NAMES = ("pt1", "pt2", "pt3", "pt4", "pt5", "pt6", "pt7")


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.dist(a, b)


def _angle_from_vertical(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Signed degrees, where 0 means the segment is vertical."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    return math.degrees(math.atan2(dx, -dy))


def _point_line_distance(point, line_start, line_end) -> float:
    p, a, b = map(np.asarray, (point, line_start, line_end))
    denominator = np.linalg.norm(b - a)
    if denominator == 0: raise ValueError("Shaft endpoints are identical")
    return float(abs(np.cross(b - a, p - a)) / denominator)


def calculate_metrics(points: dict[str, tuple[float, float]], px_per_mm: float,
                      top_coils: int, bottom_coils: int, coil_diameter_px: float) -> dict[str, float]:
    """Calculate Camera 1 dimensional metrics from named full-frame keypoints."""
    missing = set(KEYPOINT_NAMES) - points.keys()
    if missing: raise ValueError(f"Missing keypoints: {sorted(missing)}")
    mm = lambda px: px / px_per_mm
    top_height = mm(_distance(points["pt2"], points["pt3"]))
    bottom_height = mm(_distance(points["pt5"], points["pt6"]))
    return {
        "total_length": mm(_distance(points["pt1"], points["pt7"])),
        "top_coil_height": top_height,
        "bottom_coil_height": bottom_height,
        "shaft_length": mm(_distance(points["pt3"], points["pt5"])),
        "shaft_straightness": mm(_point_line_distance(points["pt4"], points["pt3"], points["pt5"])),
        "top_pitch_per_coil": top_height / top_coils,
        "bottom_pitch_per_coil": bottom_height / bottom_coils,
        "pitch_per_coil": (top_height / top_coils + bottom_height / bottom_coils) / 2,
        "coil_outer_diameter": mm(coil_diameter_px),
    }


def tolerance_failures(metrics: dict[str, float], config: dict) -> list[tuple[str, float, dict[str, float]]]:
    failures = []
    aliases = {"top_pitch_per_coil": "pitch_per_coil", "bottom_pitch_per_coil": "pitch_per_coil"}
    for metric, value in metrics.items():
        spec = config.get(metric) or config.get(aliases.get(metric, ""))
        if not spec: continue
        low, high = spec.get("min"), spec.get("max")
        max_deviation = spec.get("max_deviation", spec.get("max_degrees"))
        out = (low is not None and value < low) or (high is not None and value > high) or (max_deviation is not None and value > max_deviation)
        if out: failures.append((metric, value, spec))
    return failures
