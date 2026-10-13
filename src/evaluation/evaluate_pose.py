from pathlib import Path
import argparse
import csv
import numpy as np
from ultralytics import YOLO


KEYPOINT_NAMES = [
    "hook_tip_top",        # pt1
    "top_coil",            # pt2
    "bottom_top_coil",     # pt3
    "shaft_middle",        # pt4
    "top_bottom_coil",     # pt5
    "bottom_bottom_coil",  # pt6
    "hook_tip_bottom",     # pt7
]


def load_gt(label_path, img_width, img_height):
    """
    Read YOLO pose label.

    Format:
    class x_center y_center width height
    x1 y1 v1 x2 y2 v2 ... x7 y7 v7
    """

    with open(label_path, "r") as f:
        line = f.readline().strip()

    values = list(map(float, line.split()))

    # First 5 values = class + bbox
    keypoint_values = values[5:]

    keypoints = []

    for i in range(7):
        x = keypoint_values[i * 3]
        y = keypoint_values[i * 3 + 1]
        visibility = keypoint_values[i * 3 + 2]

        # YOLO normalized -> pixel coordinates
        px = x * img_width
        py = y * img_height

        keypoints.append([px, py, visibility])

    return np.array(keypoints)


def evaluate(model_path, data_dir, output_csv, imgsz=960, device=0):

    model = YOLO(str(model_path))

    test_images = sorted(
        (Path(data_dir) / "images").glob("*")
    )

    label_dir = Path(data_dir) / "labels"

    all_errors = [[] for _ in range(7)]
    image_results = []

    skipped = 0

    for image_path in test_images:

        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            print(f"Skipping {image_path.name}: label not found")
            skipped += 1
            continue

        # Read image size using OpenCV
        import cv2

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Skipping {image_path.name}: image could not be read")
            skipped += 1
            continue

        img_height, img_width = image.shape[:2]

        # Ground truth
        gt = load_gt(
            label_path,
            img_width,
            img_height
        )

        # Prediction
        results = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            device=device,
            verbose=False
        )

        result = results[0]

        if result.keypoints is None or len(result.keypoints.xy) == 0:
            print(f"Skipping {image_path.name}: no pose detected")
            skipped += 1
            continue

        # Select highest-confidence detected person/spring
        if result.boxes is not None and len(result.boxes.conf) > 0:

            best_idx = int(
                result.boxes.conf.argmax().item()
            )

        else:
            best_idx = 0

        pred = result.keypoints.xy[best_idx].cpu().numpy()

        # Make sure prediction has 7 keypoints
        if pred.shape[0] != 7:
            print(
                f"Skipping {image_path.name}: "
                f"expected 7 keypoints, got {pred.shape[0]}"
            )
            skipped += 1
            continue

        errors = []

        for i in range(7):

            gt_x = gt[i][0]
            gt_y = gt[i][1]

            pred_x = pred[i][0]
            pred_y = pred[i][1]

            error = np.sqrt(
                (pred_x - gt_x) ** 2 +
                (pred_y - gt_y) ** 2
            )

            errors.append(float(error))
            all_errors[i].append(float(error))

        image_mean = float(np.mean(errors))

        row = {
            "Image": image_path.name,
            "pt1": errors[0],
            "pt2": errors[1],
            "pt3": errors[2],
            "pt4": errors[3],
            "pt5": errors[4],
            "pt6": errors[5],
            "pt7": errors[6],
            "Mean": image_mean,
        }

        image_results.append(row)

        print(
            f"{image_path.name:<18} "
            f"pt1={errors[0]:6.2f} "
            f"pt2={errors[1]:6.2f} "
            f"pt3={errors[2]:6.2f} "
            f"pt4={errors[3]:6.2f} "
            f"pt5={errors[4]:6.2f} "
            f"pt6={errors[5]:6.2f} "
            f"pt7={errors[6]:6.2f} "
            f"Mean={image_mean:6.2f}"
        )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("KEYPOINT ERROR SUMMARY")
    print("=" * 75)

    print(
        f"{'Keypoint':<25}"
        f"{'Mean':>10}"
        f"{'Median':>10}"
        f"{'Min':>10}"
        f"{'Max':>10}"
        f"{'N':>8}"
    )

    print("-" * 75)

    for i, name in enumerate(KEYPOINT_NAMES):

        errors = np.array(all_errors[i])

        if len(errors) == 0:
            continue

        print(
            f"{name:<25}"
            f"{np.mean(errors):>10.2f}"
            f"{np.median(errors):>10.2f}"
            f"{np.min(errors):>10.2f}"
            f"{np.max(errors):>10.2f}"
            f"{len(errors):>8}"
        )

    # Overall
    combined_errors = np.concatenate(
        [np.array(x) for x in all_errors if len(x) > 0]
    )

    print("-" * 75)

    print(
        f"{'OVERALL':<25}"
        f"{np.mean(combined_errors):>10.2f}"
        f"{np.median(combined_errors):>10.2f}"
        f"{np.min(combined_errors):>10.2f}"
        f"{np.max(combined_errors):>10.2f}"
        f"{len(combined_errors):>8}"
    )

    print("=" * 75)

    # ---------------------------------------------------------
    # BEST / WORST IMAGES
    # ---------------------------------------------------------

    image_results.sort(key=lambda x: x["Mean"])

    print("\nBEST 5 IMAGES")
    print("-" * 50)

    for row in image_results[:5]:

        print(
            f"{row['Image']:<20}"
            f"Mean Error = {row['Mean']:.2f} px"
        )

    print("\nWORST 5 IMAGES")
    print("-" * 50)

    for row in reversed(image_results[-5:]):

        print(
            f"{row['Image']:<20}"
            f"Mean Error = {row['Mean']:.2f} px"
        )

    # ---------------------------------------------------------
    # SAVE CSV
    # ---------------------------------------------------------

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_csv,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Image",
                "pt1",
                "pt2",
                "pt3",
                "pt4",
                "pt5",
                "pt6",
                "pt7",
                "Mean",
            ],
        )

        writer.writeheader()

        for row in image_results:
            writer.writerow(row)

    print("\n")
    print(f"Evaluated : {len(image_results)}")
    print(f"Skipped   : {skipped}")
    print(f"CSV saved : {output_csv}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="D:/spring-defects/runs/pose/spring_pose_v6-2/weights/best.pt",
    )

    parser.add_argument(
        "--data",
        default="D:/spring-defects/data/yolo_pose/test",
    )

    parser.add_argument(
        "--output",
        default="D:/spring-defects/runs/evaluation/pose_error/per_image_keypoint_errors.csv",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
    )

    parser.add_argument(
        "--device",
        default="0",
    )

    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        data_dir=args.data,
        output_csv=args.output,
        imgsz=args.imgsz,
        device=args.device,
    )


if __name__ == "__main__":
    main()