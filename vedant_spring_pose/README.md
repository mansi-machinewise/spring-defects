# Spring Defect Detection - YOLO Pose

## Overview
YOLO11n-Pose model trained to detect and classify springs as good_spring or bad_spring.

## Classes
- 0: good_spring
- 1: bad_spring

## Keypoints
The model uses 4 keypoints per spring:
1. P1: Top tip of hook
2. P2: Back-side coil/upright metal section
3. P3: Base of hook where it curves toward the coil
4. P4: Center of the front-facing coil

## Files
- best.pt: Trained model weights
- data.yaml: Dataset configuration
- make_yolo_pose_dataset.py: Converts XML annotations to YOLO pose format
- check_xml.py: Validates XML annotations

## Dataset
- Training: 269 images
- Validation: 77 images
- Testing: 39 images

## Evaluation
Validation and test results were approximately 0.995 mAP@50 for both bounding boxes and pose keypoints. Results are based on the current dataset and should be validated on additional real-world images.
