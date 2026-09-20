"""Train YOLO11-pose from CVAT YOLO-pose exports with seven ordered keypoints."""
import argparse
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="YOLO pose dataset.yaml (kpt_shape must be [7, 3])")
    parser.add_argument("--base-model", default="yolo11n-pose.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--name", default="yolo_pose")
    args = parser.parse_args()
    YOLO(args.base_model).train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, project="models", name=args.name)


if __name__ == "__main__":
    main()
