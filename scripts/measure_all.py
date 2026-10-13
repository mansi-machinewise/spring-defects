import numpy as np
from ultralytics import YOLO
from pathlib import Path

PIXELS_PER_MM = 20.6978

model = YOLO('D:/spring-defects/runs/pose/spring_pose_v6-2/weights/best.pt')

train_dir = Path('D:/spring-defects/data/yolo_pose/train/images')
images = sorted(train_dir.glob('*.jpeg'))

total_lengths = []
hook_lengths  = []

print(f'{"Image":<20} {"Total(mm)":>10} {"Hook(mm)":>10}')
print('-' * 42)

for img_path in images:
    res = model.predict(source=str(img_path), imgsz=960, conf=0.25, device=0, verbose=False)[0]
    if res.keypoints is None or len(res.keypoints.data) == 0:
        print(f'{img_path.name:<20} NO DETECTION')
        continue
    kd = res.keypoints.data.cpu().numpy()[0]
    kpt0, kpt1, kpt5, kpt6 = kd[0,:2], kd[1,:2], kd[5,:2], kd[6,:2]
    total_mm = float(np.sqrt((kpt0[0]-kpt6[0])**2 + (kpt0[1]-kpt6[1])**2)) / PIXELS_PER_MM
    hook_mm  = float(np.sqrt((kpt1[0]-kpt5[0])**2 + (kpt1[1]-kpt5[1])**2)) / PIXELS_PER_MM
    total_lengths.append(total_mm)
    hook_lengths.append(hook_mm)
    print(f'{img_path.name:<20} {total_mm:>10.2f} {hook_mm:>10.2f}')

print('-' * 42)
print(f'\nTotal length statistics:')
print(f'  Mean   : {np.mean(total_lengths):.2f} mm')
print(f'  Std    : {np.std(total_lengths):.2f} mm')
print(f'  Min    : {np.min(total_lengths):.2f} mm')
print(f'  Max    : {np.max(total_lengths):.2f} mm')
print(f'  Range  : {np.max(total_lengths)-np.min(total_lengths):.2f} mm')

print(f'\nHook length statistics:')
print(f'  Mean   : {np.mean(hook_lengths):.2f} mm')
print(f'  Std    : {np.std(hook_lengths):.2f} mm')
print(f'  Min    : {np.min(hook_lengths):.2f} mm')
print(f'  Max    : {np.max(hook_lengths):.2f} mm')
print(f'  Range  : {np.max(hook_lengths)-np.min(hook_lengths):.2f} mm')
