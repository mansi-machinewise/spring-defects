import numpy as np
from ultralytics import YOLO
from pathlib import Path

model = YOLO('D:/spring-defects/runs/pose/spring_pose_v6-2/weights/best.pt')
train_dir = Path('D:/spring-defects/data/yolo_pose/train/images')

good_confs = []
bad_confs  = []
good_scores = []
bad_scores  = []

for img_path in sorted(train_dir.glob('*.jpeg')):
    num = int(img_path.stem.split('_')[1])
    res = model.predict(source=str(img_path), imgsz=960, conf=0.25, device=0, verbose=False)[0]
    if res.keypoints is None or len(res.keypoints.data) == 0:
        continue
    kd = res.keypoints.data.cpu().numpy()[0]
    confs = kd[:, 2]
    mean_conf = float(np.mean(confs))
    min_conf  = float(np.min(confs))

    if num <= 100:
        good_confs.append(mean_conf)
        good_scores.append(min_conf)
    else:
        bad_confs.append(mean_conf)
        bad_scores.append(min_conf)

print('GOOD springs keypoint confidence:')
print(f'  Mean confidence : {np.mean(good_confs):.4f}')
print(f'  Std             : {np.std(good_confs):.4f}')
print(f'  Min             : {np.min(good_confs):.4f}')
print(f'  Max             : {np.max(good_confs):.4f}')
print(f'  Mean min-conf   : {np.mean(good_scores):.4f}')

print()
print('BAD springs keypoint confidence:')
print(f'  Mean confidence : {np.mean(bad_confs):.4f}')
print(f'  Std             : {np.std(bad_confs):.4f}')
print(f'  Min             : {np.min(bad_confs):.4f}')
print(f'  Max             : {np.max(bad_confs):.4f}')
print(f'  Mean min-conf   : {np.mean(bad_scores):.4f}')

print()
print(f'Confidence difference: {np.mean(good_confs) - np.mean(bad_confs):.4f}')
