import numpy as np
from ultralytics import YOLO
from pathlib import Path

model = YOLO('D:/spring-defects/runs/pose/spring_pose_v6-2/weights/best.pt')
train_dir = Path('D:/spring-defects/data/yolo_pose/train/images')

good_offsets, bad_offsets = [], []
good_tilts,   bad_tilts   = [], []
good_shafts,  bad_shafts  = [], []

print(f'{"Image":<20} {"Hook_offset":>12} {"Tilt(deg)":>10} {"Shaft_dev":>10} {"Label":>6}')
print('-' * 65)

for img_path in sorted(train_dir.glob('*.jpeg')):
    num = int(img_path.stem.split('_')[1])
    label = 'GOOD' if num <= 100 else 'BAD'

    res = model.predict(source=str(img_path), imgsz=960, conf=0.25, device=0, verbose=False)[0]
    if res.keypoints is None or len(res.keypoints.data) == 0:
        continue

    kd = res.keypoints.data.cpu().numpy()[0]
    kpt0 = kd[0,:2]
    kpt2 = kd[2,:2]
    kpt3 = kd[3,:2]
    kpt4 = kd[4,:2]
    kpt6 = kd[6,:2]

    hook_offset = abs(kpt0[0] - kpt6[0])
    dx = kpt6[0] - kpt0[0]
    dy = kpt6[1] - kpt0[1]
    tilt_angle = abs(np.degrees(np.arctan2(dx, dy)))
    x1,y1 = kpt2; x2,y2 = kpt4; x0,y0 = kpt3
    shaft_dev = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2-x1)**2)

    if label == 'GOOD':
        good_offsets.append(hook_offset)
        good_tilts.append(tilt_angle)
        good_shafts.append(shaft_dev)
    else:
        bad_offsets.append(hook_offset)
        bad_tilts.append(tilt_angle)
        bad_shafts.append(shaft_dev)

    print(f'{img_path.name:<20} {hook_offset:>12.1f} {tilt_angle:>10.2f} {shaft_dev:>10.1f} {label:>6}')

print('\n' + '='*65)
print(f'{"Metric":<20} {"GOOD mean":>12} {"GOOD max":>10} {"BAD mean":>10} {"BAD min":>8}')
print('-'*65)
print(f'{"Hook offset (px)":<20} {np.mean(good_offsets):>12.1f} {np.max(good_offsets):>10.1f} {np.mean(bad_offsets):>10.1f} {np.min(bad_offsets):>8.1f}')
print(f'{"Tilt angle (deg)":<20} {np.mean(good_tilts):>12.2f} {np.max(good_tilts):>10.2f} {np.mean(bad_tilts):>10.2f} {np.min(bad_tilts):>8.2f}')
print(f'{"Shaft dev (px)":<20} {np.mean(good_shafts):>12.1f} {np.max(good_shafts):>10.1f} {np.mean(bad_shafts):>10.1f} {np.min(bad_shafts):>8.1f}')
