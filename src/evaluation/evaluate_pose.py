"""
Quantitative keypoint pixel-error evaluation for YOLO11-pose spring model.

Evaluates all test images and reports per-keypoint Euclidean pixel error
(predicted vs ground-truth), with an overall summary.

Output saved to:
    D:/spring-defects/runs/evaluation/pose_error/
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

# ── Constants ─────────────────────────────────────────────────────────────────

KEYPOINT_NAMES = [
    "hook_tip_top",
    "top_coil",
    "bottom_top_coil",
    "shaft_middle",
    "top_bottom_coil",
    "bottom_bottom_coil",
    "hook_tip_bottom",
]
NUM_KPT = len(KEYPOINT_NAMES)

# ── Helpers ───────────────────────────────────────────────────────────────────


def load_yaml(yaml_path: Path) -> dict:
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def parse_label_file(label_path: Path, img_w: int, img_h: int):
    """
    Parse a single YOLO-pose label file.

    YOLO pose format (one line per object):
        class cx cy bw bh  x0 y0 v0  x1 y1 v1 ... x6 y6 v6
    All values normalised [0, 1].

    Returns list of dicts, one per annotated spring:
        {
            "cx": float, "cy": float, "bw": float, "bh": float,
            "keypoints": [(px_x, px_y, vis), ...] length 7
        }
    """
    objects = []
    if not label_path.exists():
        return objects

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            # class + box (5) + 7 kpts * 3 = 26 values minimum
            if len(parts) < 5 + NUM_KPT * 3:
                continue
            # class_id = int(parts[0])  # always 0 for spring
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            kpt_raw = parts[5:]
            keypoints = []
            for i in range(NUM_KPT):
                xn = float(kpt_raw[i * 3])
                yn = float(kpt_raw[i * 3 + 1])
                vis = float(kpt_raw[i * 3 + 2])
                keypoints.append((xn * img_w, yn * img_h, vis))
            objects.append(
                {
                    "cx": cx * img_w,
                    "cy": cy * img_h,
                    "bw": bw * img_w,
                    "bh": bh * img_h,
                    "keypoints": keypoints,
                }
            )
    return objects


def euclidean(p1, p2):
    return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))


def match_prediction_to_gt(pred_cx, pred_cy, gt_objects):
    """
    Match the closest ground-truth object (by box centre distance) to a
    single prediction.  With one spring per image this is trivial.
    """
    if not gt_objects:
        return None
    dists = [
        euclidean((pred_cx, pred_cy), (o["cx"], o["cy"])) for o in gt_objects
    ]
    return gt_objects[int(np.argmin(dists))]


# ── Main evaluation ───────────────────────────────────────────────────────────


def evaluate(
    model_path: Path,
    data_yaml: Path,
    output_dir: Path,
    conf_threshold: float = 0.25,
    imgsz: int = 960,
    device: str = "0",
):
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(data_yaml)
    dataset_root = Path(cfg["path"])
    test_images_dir = dataset_root / cfg.get("test", "test/images")
    test_labels_dir = test_images_dir.parent.parent / "test" / "labels"

    print(f"\nModel   : {model_path}")
    print(f"Images  : {test_images_dir}")
    print(f"Labels  : {test_labels_dir}")
    print(f"Output  : {output_dir}\n")

    model = YOLO(str(model_path))

    image_paths = sorted(
        p
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        for p in test_images_dir.glob(ext)
    )

    if not image_paths:
        print(f"ERROR: No images found in {test_images_dir}")
        return

    print(f"Found {len(image_paths)} test images.\n")

    # per-keypoint errors: list of pixel errors across all evaluated instances
    kpt_errors: list[list[float]] = [[] for _ in range(NUM_KPT)]

    per_image_results = []
    skipped = 0

    for img_path in image_paths:
        label_path = test_labels_dir / (img_path.stem + ".txt")

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] Cannot read image: {img_path.name}")
            skipped += 1
            continue

        img_h, img_w = img.shape[:2]
        gt_objects = parse_label_file(label_path, img_w, img_h)

        if not gt_objects:
            print(f"  [WARN] No ground-truth labels for: {img_path.name}")
            skipped += 1
            continue

        results = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=conf_threshold,
            device=device,
            verbose=False,
        )

        result = results[0]

        if result.keypoints is None or len(result.keypoints.data) == 0:
            print(f"  [WARN] No detection in: {img_path.name}")
            skipped += 1
            continue

        # Take the highest-confidence box
        if result.boxes is not None and len(result.boxes) > 0:
            confs = result.boxes.conf.cpu().numpy()
            best_idx = int(np.argmax(confs))
        else:
            best_idx = 0

        kpts_data = result.keypoints.data.cpu().numpy()[best_idx]  # (7, 3)

        # Centre of predicted box for matching
        box = result.boxes.xywh.cpu().numpy()[best_idx]
        pred_cx, pred_cy = box[0], box[1]

        gt = match_prediction_to_gt(pred_cx, pred_cy, gt_objects)
        if gt is None:
            skipped += 1
            continue

        img_errors = []
        for ki in range(NUM_KPT):
            px, py, pconf = kpts_data[ki]
            gx, gy, gvis = gt["keypoints"][ki]
            err = euclidean((px, py), (gx, gy))
            kpt_errors[ki].append(err)
            img_errors.append(
                {
                    "name": KEYPOINT_NAMES[ki],
                    "pred_x": round(float(px), 1),
                    "pred_y": round(float(py), 1),
                    "pred_conf": round(float(pconf), 4),
                    "gt_x": round(gx, 1),
                    "gt_y": round(gy, 1),
                    "error_px": round(err, 2),
                }
            )

        per_image_results.append(
            {"image": img_path.name, "keypoints": img_errors}
        )
        overall_err = float(np.mean([e["error_px"] for e in img_errors]))
        print(
            f"  {img_path.name:<35}  mean_err={overall_err:7.2f} px"
        )

    evaluated = len(per_image_results)
    print(f"\nEvaluated: {evaluated}  |  Skipped: {skipped}\n")

    if evaluated == 0:
        print("No images evaluated – cannot produce report.")
        return

    # ── Per-keypoint statistics ───────────────────────────────────────────────
    header = f"{'Keypoint':<22}  {'Mean':>8}  {'Median':>8}  {'Min':>8}  {'Max':>8}  {'N':>4}"
    separator = "-" * len(header)

    lines = [header, separator]
    all_errors = []
    kpt_stats = []

    for ki, name in enumerate(KEYPOINT_NAMES):
        errs = kpt_errors[ki]
        all_errors.extend(errs)
        if errs:
            mean_e = float(np.mean(errs))
            med_e = float(np.median(errs))
            min_e = float(np.min(errs))
            max_e = float(np.max(errs))
            n = len(errs)
        else:
            mean_e = med_e = min_e = max_e = float("nan")
            n = 0

        kpt_stats.append(
            {
                "keypoint": name,
                "mean_px": round(mean_e, 2),
                "median_px": round(med_e, 2),
                "min_px": round(min_e, 2),
                "max_px": round(max_e, 2),
                "n": n,
            }
        )
        lines.append(
            f"{name:<22}  {mean_e:>8.2f}  {med_e:>8.2f}  {min_e:>8.2f}  {max_e:>8.2f}  {n:>4}"
        )

    lines.append(separator)
    overall_mean = float(np.mean(all_errors)) if all_errors else float("nan")
    overall_med = float(np.median(all_errors)) if all_errors else float("nan")
    overall_min = float(np.min(all_errors)) if all_errors else float("nan")
    overall_max = float(np.max(all_errors)) if all_errors else float("nan")
    lines.append(
        f"{'OVERALL':<22}  {overall_mean:>8.2f}  {overall_med:>8.2f}  "
        f"{overall_min:>8.2f}  {overall_max:>8.2f}  {len(all_errors):>4}"
    )
    lines.append("")
    lines.append(f"Images evaluated : {evaluated}")
    lines.append(f"Images skipped   : {skipped}")

    report_text = "\n".join(lines)
    print("\n" + report_text)

    # ── Save outputs ─────────────────────────────────────────────────────────
    report_path = output_dir / "keypoint_error_report.txt"
    report_path.write_text(report_text)
    print(f"\nReport saved : {report_path}")

    csv_path = output_dir / "keypoint_stats.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=kpt_stats[0].keys())
        w.writeheader()
        w.writerows(kpt_stats)
    print(f"CSV saved    : {csv_path}")

    json_path = output_dir / "per_image_results.json"
    with open(json_path, "w") as f:
        json.dump(per_image_results, f, indent=2)
    print(f"JSON saved   : {json_path}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO11-pose spring keypoint pixel accuracy."
    )
    parser.add_argument(
        "--model",
        default="D:/spring-defects/runs/pose/spring_pose_v1/weights/best.pt",
    )
    parser.add_argument(
        "--data",
        default="D:/spring-defects/config/spring_pose.yaml",
    )
    parser.add_argument(
        "--output",
        default="D:/spring-defects/runs/evaluation/pose_error",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")

    args = parser.parse_args()

    evaluate(
        model_path=Path(args.model),
        data_yaml=Path(args.data),
        output_dir=Path(args.output),
        conf_threshold=args.conf,
        imgsz=args.imgsz,
        device=args.device,
    )