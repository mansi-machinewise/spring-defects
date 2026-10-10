from pathlib import Path
import csv

from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "classify"
    / "spring_cls_v4"
    / "weights"
    / "best.pt"
)

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "yolo_cls"
    / "test"
    / "lighting_eval"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "runs"
    / "classify"
    / "spring_cls_v4"
    / "lighting_confidence_results.csv"
)


# ============================================================
# SETTINGS
# ============================================================

CONDITIONS = [
    "normal",
    "slightly_dark",
    "slightly_bright",
    "low_contrast",
    "high_contrast",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# YOLO class mapping
DEFECTIVE_CLASS = 0
GOOD_CLASS = 1


# ============================================================
# GET IMAGES
# ============================================================

def get_images(folder):
    return sorted(
        [
            p
            for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("YOLO11s V4 - LIGHTING CONFIDENCE ANALYSIS")
    print("=" * 75)

    print(f"Model:   {MODEL_PATH}")
    print(f"Dataset: {DATASET_ROOT}")
    print()

    # Load model
    model = YOLO(str(MODEL_PATH))

    rows = []

    # --------------------------------------------------------
    # Evaluate every condition
    # --------------------------------------------------------

    for condition in CONDITIONS:

        print(f"Processing: {condition}")

        good_folder = DATASET_ROOT / "good" / condition
        defective_folder = DATASET_ROOT / "defective" / condition

        # -------------------------
        # GOOD images
        # -------------------------

        good_images = get_images(good_folder)

        if good_images:
            results = model.predict(
                source=[str(p) for p in good_images],
                verbose=False,
            )

            for image_path, result in zip(good_images, results):

                predicted_class = int(result.probs.top1)
                confidence = float(result.probs.top1conf)

                correct = predicted_class == GOOD_CLASS

                rows.append(
                    {
                        "filename": image_path.name,
                        "actual": "GOOD",
                        "predicted": (
                            "GOOD"
                            if predicted_class == GOOD_CLASS
                            else "DEFECTIVE"
                        ),
                        "confidence": confidence,
                        "condition": condition,
                        "correct": correct,
                    }
                )

        # -------------------------
        # DEFECTIVE images
        # -------------------------

        defective_images = get_images(defective_folder)

        if defective_images:
            results = model.predict(
                source=[str(p) for p in defective_images],
                verbose=False,
            )

            for image_path, result in zip(defective_images, results):

                predicted_class = int(result.probs.top1)
                confidence = float(result.probs.top1conf)

                correct = predicted_class == DEFECTIVE_CLASS

                rows.append(
                    {
                        "filename": image_path.name,
                        "actual": "DEFECTIVE",
                        "predicted": (
                            "GOOD"
                            if predicted_class == GOOD_CLASS
                            else "DEFECTIVE"
                        ),
                        "confidence": confidence,
                        "condition": condition,
                        "correct": correct,
                    }
                )

    # ========================================================
    # SAVE CSV
    # ========================================================

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "filename",
                "actual",
                "predicted",
                "confidence",
                "condition",
                "correct",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    # ========================================================
    # SUMMARY
    # ========================================================

    correct_rows = [
        row for row in rows
        if row["correct"]
    ]

    incorrect_rows = [
        row for row in rows
        if not row["correct"]
    ]

    correct_confidences = [
        float(row["confidence"])
        for row in correct_rows
    ]

    incorrect_confidences = [
        float(row["confidence"])
        for row in incorrect_rows
    ]

    print()
    print("=" * 75)
    print("CONFIDENCE SUMMARY")
    print("=" * 75)

    print(f"Total images:       {len(rows)}")
    print(f"Correct:             {len(correct_rows)}")
    print(f"Incorrect:           {len(incorrect_rows)}")

    if correct_confidences:
        print()
        print("CORRECT PREDICTIONS")
        print("-" * 40)
        print(
            f"Minimum confidence: {min(correct_confidences):.4f}"
        )
        print(
            f"Maximum confidence: {max(correct_confidences):.4f}"
        )
        print(
            f"Average confidence: {sum(correct_confidences) / len(correct_confidences):.4f}"
        )

    if incorrect_confidences:
        print()
        print("INCORRECT PREDICTIONS")
        print("-" * 40)

        for row in incorrect_rows:
            print(
                f"{row['filename']} | "
                f"Actual: {row['actual']} | "
                f"Predicted: {row['predicted']} | "
                f"Confidence: {float(row['confidence']):.4f} | "
                f"Condition: {row['condition']}"
            )

    print()
    print("=" * 75)
    print(f"Detailed results saved to:")
    print(OUTPUT_CSV)
    print("=" * 75)


if __name__ == "__main__":
    main()