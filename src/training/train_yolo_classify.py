"""
Train YOLO11n-cls on spring images for GOOD/BAD classification.

Dataset:
    train: 200 GOOD + 200 BAD
    val:   50 GOOD + 50 BAD

Output:
    D:/spring-defects/runs/classify/spring_cls_v1/weights/best.pt

Usage:
    D:\\spring-defects\\.venv\\Scripts\\python.exe D:\\spring-defects\\src\\training\\train_yolo_classify.py
"""

from ultralytics import YOLO


def main() -> None:
    data       = "D:/spring-defects/data/classify"
    base_model = "yolo11n-cls.pt"
    epochs     = 100
    imgsz      = 960
    batch      = 8
    device     = "0"
    name       = "spring_cls_v1"

    YOLO(base_model).train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="D:/spring-defects/runs/classify",
        name=name,
        patience=20,
        optimizer="AdamW",
        cos_lr=True,
        save_period=20,
        val=True,
        verbose=True,
        workers=0,
    )


if __name__ == "__main__":
    main()