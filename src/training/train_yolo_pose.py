"""Train YOLO11-pose from CVAT YOLO-pose exports with seven ordered keypoints."""

from ultralytics import YOLO


def main() -> None:
    data       = "D:/spring-defects/config/spring_pose.yaml"
    base_model = "yolo11n-pose.pt"
    epochs     = 300
    imgsz      = 1280
    batch      = 16
    device     = "0"
    name       = "spring_pose_v7"


    YOLO(base_model).train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="D:/spring-defects/runs/pose",
        name=name,
        pose=12.0,
        kobj=2.0,
        patience=100,
        optimizer="AdamW",
        cos_lr=True,
        save_period=50,
        val=True,
        verbose=True,
        fliplr=0.0,
        cache=True,
        workers=0,
    )


if __name__ == "__main__":
    main()