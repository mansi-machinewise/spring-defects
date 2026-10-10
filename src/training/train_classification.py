from pathlib import Path

import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "classification_v4.yaml"


def main():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model = YOLO(config["model"])

    augmentation = config.get("augmentation", {})

    model.train(
        data=str(PROJECT_ROOT / config["data"]),
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        device=config["device"],
        pretrained=config["pretrained"],
        project=str(PROJECT_ROOT / "runs" / "classify"),
        name=config["experiment"],

        # Classification augmentation
        fliplr=augmentation.get("fliplr", 0.5),
        translate=augmentation.get("translate", 0.1),
        scale=augmentation.get("scale", 0.5),
        hsv_h=augmentation.get("hsv_h", 0.015),
        hsv_s=augmentation.get("hsv_s", 0.7),
        hsv_v=augmentation.get("hsv_v", 0.4),
        auto_augment=augmentation.get("auto_augment", "randaugment"),
        erasing=augmentation.get("erasing", 0.4),
    )


if __name__ == "__main__":
    main()