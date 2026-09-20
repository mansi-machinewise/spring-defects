"""Train YOLO11 spring-presence detection from CVAT YOLO-detect exports."""
import argparse
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="YOLO dataset.yaml for spring bounding boxes")
    parser.add_argument("--base-model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    args = parser.parse_args()
    YOLO(args.base_model).train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, project="models", name="yolo_detect")


if __name__ == "__main__":
    main()
