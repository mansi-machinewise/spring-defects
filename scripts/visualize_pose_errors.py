from pathlib import Path
import csv
import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
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

CSV_PATH = Path(
    r"D:\spring-defects\runs\evaluation\pose_error\per_image_keypoint_errors.csv"
)

OUTPUT_DIR = Path(
    r"D:\spring-defects\runs\evaluation\pose_error\visualizations"
)

IMG_SIZE = 960
DEVICE = 0

# Number of worst images to visualize
TOP_N = 10


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
# READ GROUND TRUTH
# ============================================================

def load_gt(label_path, img_width, img_height):

    with open(label_path, "r") as f:
        line = f.readline().strip()

    values = list(map(float, line.split()))

    # First 5:
    # class x_center y_center width height
    keypoint_values = values[5:]

    keypoints = []

    for i in range(7):

        x = keypoint_values[i * 3]
        y = keypoint_values[i * 3 + 1]

        px = x * img_width
        py = y * img_height

        keypoints.append([px, py])

    return np.array(keypoints, dtype=np.float32)


# ============================================================
# DRAW KEYPOINTS
# ============================================================

def draw_keypoints(
    image,
    gt,
    pred,
    errors
):

    output = image.copy()

    for i in range(7):

        gt_x = int(round(gt[i][0]))
        gt_y = int(round(gt[i][1]))

        pred_x = int(round(pred[i][0]))
        pred_y = int(round(pred[i][1]))

        error = errors[i]

        # ----------------------------------------------------
        # Ground truth = GREEN
        # ----------------------------------------------------

        cv2.circle(
            output,
            (gt_x, gt_y),
            8,
            (0, 255, 0),
            -1
        )

        # ----------------------------------------------------
        # Prediction = RED
        # ----------------------------------------------------

        cv2.circle(
            output,
            (pred_x, pred_y),
            8,
            (0, 0, 255),
            -1
        )

        # ----------------------------------------------------
        # GT -> Prediction line
        # ----------------------------------------------------

        cv2.line(
            output,
            (gt_x, gt_y),
            (pred_x, pred_y),
            (255, 0, 255),
            2
        )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label = f"{KEYPOINT_NAMES[i]}: {error:.1f}px"

        text_x = pred_x + 10
        text_y = pred_y - 10

        # Keep text inside image
        text_x = max(5, min(text_x, output.shape[1] - 250))
        text_y = max(20, min(text_y, output.shape[0] - 10))

        cv2.putText(
            output,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )

    return output


# ============================================================
# ADD HEADER
# ============================================================

def add_header(image, image_name, mean_error):

    header_height = 100

    header = np.zeros(
        (header_height, image.shape[1], 3),
        dtype=np.uint8
    )

    cv2.putText(
        header,
        image_name,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        header,
        f"Mean Keypoint Error: {mean_error:.2f} px",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    return np.vstack([header, image])


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV not found:\n{CSV_PATH}"
        )

    # --------------------------------------------------------
    # Read CSV and select worst images
    # --------------------------------------------------------

    rows = []

    with open(
        CSV_PATH,
        "r",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            row["Mean"] = float(row["Mean"])

            rows.append(row)

    rows.sort(
        key=lambda x: x["Mean"],
        reverse=True
    )

    worst_images = rows[:TOP_N]

    print("\nWorst images selected:")
    print("-" * 50)

    for row in worst_images:

        print(
            f"{row['Image']:<20}"
            f"{row['Mean']:.2f} px"
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading model...")

    model = YOLO(str(MODEL_PATH))

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    print("\nGenerating visualizations...\n")

    for rank, row in enumerate(worst_images, start=1):

        image_name = row["Image"]

        image_path = TEST_IMAGES / image_name

        label_path = TEST_LABELS / (
            Path(image_name).stem + ".txt"
        )

        if not image_path.exists():

            print(
                f"Skipping {image_name}: "
                "image not found"
            )

            continue

        if not label_path.exists():

            print(
                f"Skipping {image_name}: "
                "label not found"
            )

            continue

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                f"Skipping {image_name}: "
                "could not read image"
            )

            continue

        height, width = image.shape[:2]

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        gt = load_gt(
            label_path,
            width,
            height
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        results = model.predict(
            source=str(image_path),
            imgsz=IMG_SIZE,
            device=DEVICE,
            verbose=False
        )

        result = results[0]

        if (
            result.keypoints is None
            or len(result.keypoints.xy) == 0
        ):

            print(
                f"Skipping {image_name}: "
                "no prediction"
            )

            continue

        # Highest-confidence detection
        if (
            result.boxes is not None
            and len(result.boxes.conf) > 0
        ):

            best_idx = int(
                result.boxes.conf.argmax().item()
            )

        else:

            best_idx = 0

        pred = (
            result
            .keypoints
            .xy[best_idx]
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Calculate errors again
        # ----------------------------------------------------

        errors = []

        for i in range(7):

            dx = pred[i][0] - gt[i][0]
            dy = pred[i][1] - gt[i][1]

            error = np.sqrt(
                dx ** 2 + dy ** 2
            )

            errors.append(float(error))

        mean_error = float(
            np.mean(errors)
        )

        # ----------------------------------------------------
        # Draw
        # ----------------------------------------------------

        annotated = draw_keypoints(
            image,
            gt,
            pred,
            errors
        )

        annotated = add_header(
            annotated,
            image_name,
            mean_error
        )

        # ----------------------------------------------------
        # Legend
        # ----------------------------------------------------

        cv2.circle(
            annotated,
            (30, 125),
            8,
            (0, 255, 0),
            -1
        )

        cv2.putText(
            annotated,
            "GREEN = Ground Truth",
            (50, 132),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.circle(
            annotated,
            (300, 125),
            8,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            annotated,
            "RED = Prediction",
            (320, 132),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        output_name = (
            f"{rank:02d}_{Path(image_name).stem}_"
            f"error_{mean_error:.2f}px.jpg"
        )

        output_path = OUTPUT_DIR / output_name

        cv2.imwrite(
            str(output_path),
            annotated
        )

        print(
            f"[{rank}/{TOP_N}] "
            f"{image_name} -> "
            f"{output_path}"
        )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    print(
        f"\nVisualizations saved to:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()