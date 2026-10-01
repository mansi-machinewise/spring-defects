"""Train YOLO11-pose from CVAT YOLO-pose exports with seven ordered keypoints."""

from ultralytics import YOLO


def main() -> None:
    data       = "D:/spring-defects/config/spring_pose.yaml"
    base_model = "yolo11n-pose.pt"
    epochs     = 150
    imgsz      = 960
    batch      = 8
    device     = "0"
    name       = "spring_pose_v4"

    YOLO(base_model).train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="D:/spring-defects/runs/pose",
        name=name,

        # --- pose specific ---
        pose=12.0,
        kobj=2.0,
        patience=50,
        optimizer="AdamW",
        cos_lr=True,
        save_period=50,

        # --- augmentation ---
        fliplr=0.0,
        flipud=0.0,
        degrees=8,
        scale=0.05,
        translate=0.1,
        hsv_v=0.3,
        hsv_s=0.3,
        mosaic=0.0,
        mixup=0.0,

        # --- misc ---
        val=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()