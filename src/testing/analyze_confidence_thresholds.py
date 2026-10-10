from pathlib import Path
import csv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    PROJECT_ROOT
    / "runs"
    / "classify"
    / "spring_cls_v4"
    / "lighting_confidence_results.csv"
)


THRESHOLDS = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]


def main():

    print("=" * 100)
    print("YOLO11s V4 - CONFIDENCE THRESHOLD ANALYSIS")
    print("=" * 100)

    rows = []

    with open(CSV_PATH, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            row["confidence"] = float(row["confidence"])
            row["correct"] = row["correct"].lower() == "true"
            rows.append(row)

    print(f"Total images: {len(rows)}")
    print()

    print(
        f"{'Threshold':<12}"
        f"{'ACCEPT':<12}"
        f"{'REJECT':<12}"
        f"{'UNCERTAIN':<14}"
        f"{'Errors Caught':<15}"
        f"{'Correct Made Uncertain':<25}"
    )

    print("-" * 100)

    for threshold in THRESHOLDS:

        accepted = 0
        rejected = 0
        uncertain = 0

        errors_caught = 0
        correct_made_uncertain = 0

        for row in rows:

            confidence = row["confidence"]

            if confidence < threshold:

                uncertain += 1

                if row["correct"]:
                    correct_made_uncertain += 1
                else:
                    errors_caught += 1

            else:

                if row["predicted"] == "GOOD":
                    accepted += 1
                else:
                    rejected += 1

        print(
            f"{threshold:<12.2f}"
            f"{accepted:<12}"
            f"{rejected:<12}"
            f"{uncertain:<14}"
            f"{errors_caught:<15}"
            f"{correct_made_uncertain:<25}"
        )

    print()
    print("=" * 100)
    print("KNOWN INCORRECT PREDICTIONS")
    print("=" * 100)

    for row in rows:

        if not row["correct"]:

            print(
                f"{row['filename']} | "
                f"Actual: {row['actual']} | "
                f"Predicted: {row['predicted']} | "
                f"Confidence: {row['confidence']:.4f} | "
                f"Condition: {row['condition']}"
            )

    print()
    print("=" * 100)


if __name__ == "__main__":
    main()