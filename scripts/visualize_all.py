from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = Path(
    r"D:\spring-defects\runs\pose\spring_pose_v8-3\weights\best.pt"
)

TEST_IMAGES = Path(
    r"D:\spring-defects\data\yolo_pose\test\images"
)

TEST_LABELS = Path(
    r"D:\spring-defects\data\yolo_pose\test\labels"
)

OUTPUT_DIR = Path(
    r"D:\spring-defects\runs\evaluation\pose_error\all_visualizations"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 960
DEVICE = 0

KEYPOINT_NAMES = [
    "pt1",
    "pt2",
    "pt3",
    "pt4",
    "pt5",
    "pt6",
    "pt7",
]


# ============================================================
# READ YOLO POSE LABEL
# ============================================================

def read_gt_keypoints(label_path: Path, width: int, height: int):
    """
    YOLO pose format:

    class
    bbox_x
    bbox_y
    bbox_w
    bbox_h
    pt1_x pt1_y pt1_visible
    ...
    pt7_x pt7_y pt7_visible
    """

    if not label_path.exists():
        return None

    lines = label_path.read_text().strip().splitlines()

    if not lines:
        return None

    # Use first object
    values = lines[0].split()

    # 5 bbox values + 21 keypoint values = 26
    if len(values) < 26:
        print(f"WARNING: Invalid label: {label_path}")
        return None

    keypoints = []

    for i in range(7):
        x = float(values[5 + i * 3])
        y = float(values[5 + i * 3 + 1])
        visible = float(values[5 + i * 3 + 2])

        px = x * width
        py = y * height

        keypoints.append([px, py, visible])

    return np.array(keypoints, dtype=np.float32)


# ============================================================
# DRAW KEYPOINTS
# ============================================================

def draw_keypoint(
    image,
    point,
    color,
    label,
    radius=7,
):
    x, y = int(point[0]), int(point[1])

    cv2.circle(
        image,
        (x, y),
        radius,
        color,
        -1,
    )

    cv2.putText(
        image,
        label,
        (x + 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("ALL GOOD + BAD IMAGE VISUALIZATION")
    print("=" * 60)

    print(f"Model:  {MODEL_PATH}")
    print(f"Images: {TEST_IMAGES}")
    print(f"Labels: {TEST_LABELS}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("Loading V8-3 model...")

    model = YOLO(str(MODEL_PATH))

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    image_files = sorted(
        [
            p
            for p in TEST_IMAGES.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
        ]
    )

    print(f"Total images found: {len(image_files)}")
    print()

    if not image_files:
        print("No images found.")
        return

    # --------------------------------------------------------
    # Process every image
    # --------------------------------------------------------

    for index, image_path in enumerate(image_files, start=1):

        print(
            f"[{index}/{len(image_files)}] "
            f"Processing {image_path.name}"
        )

        image = cv2.imread(str(image_path))

        if image is None:
            print("  WARNING: Could not read image.")
            continue

        height, width = image.shape[:2]

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        label_path = TEST_LABELS / f"{image_path.stem}.txt"

        gt = read_gt_keypoints(
            label_path,
            width,
            height,
        )

        if gt is None:
            print("  WARNING: Ground-truth label missing/invalid.")
            continue

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        results = model.predict(
            source=str(image_path),
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            verbose=False,
        )

        result = results[0]

        if (
            result.keypoints is None
            or result.keypoints.xy is None
            or len(result.keypoints.xy) == 0
        ):
            print("  WARNING: No pose detected.")
            continue

        # ----------------------------------------------------
        # Select highest confidence detection
        # ----------------------------------------------------

        if result.boxes is not None and result.boxes.conf is not None:
            confidences = result.boxes.conf.cpu().numpy()

            best_index = int(
                np.argmax(confidences)
            )
        else:
            best_index = 0

        pred = (
            result.keypoints.xy[best_index]
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Calculate errors
        # ----------------------------------------------------

        errors = np.linalg.norm(
            gt[:, :2] - pred,
            axis=1,
        )

        mean_error = float(np.mean(errors))

        # ----------------------------------------------------
        # Draw
        # ----------------------------------------------------

        output = image.copy()

        # GREEN = Ground Truth
        # RED = Prediction
        # MAGENTA = error line

        for i in range(7):

            gt_point = gt[i, :2]
            pred_point = pred[i]

            gt_xy = (
                int(gt_point[0]),
                int(gt_point[1]),
            )

            pred_xy = (
                int(pred_point[0]),
                int(pred_point[1]),
            )

            # GT -> Prediction line
            cv2.line(
                output,
                gt_xy,
                pred_xy,
                (255, 0, 255),
                2,
            )

            # Ground Truth - GREEN
            draw_keypoint(
                output,
                gt_point,
                (0, 255, 0),
                f"{KEYPOINT_NAMES[i]} GT",
            )

            # Prediction - RED
            draw_keypoint(
                output,
                pred_point,
                (0, 0, 255),
                f"{KEYPOINT_NAMES[i]} {errors[i]:.1f}px",
            )

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        header_height = 80

        canvas = np.zeros(
            (
                height + header_height,
                width,
                3,
            ),
            dtype=np.uint8,
        )

        canvas[
            header_height:,
            :
        ] = output

        cv2.putText(
            canvas,
            image_path.name,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            f"Mean Keypoint Error: {mean_error:.2f}px",
            (20, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        output_path = (
            OUTPUT_DIR
            / f"{index:03d}_{image_path.stem}_"
              f"error_{mean_error:.2f}px.jpg"
        )

        cv2.imwrite(
            str(output_path),
            canvas,
        )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print()
    print(f"Visualizations saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()