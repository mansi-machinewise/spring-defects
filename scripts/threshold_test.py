import numpy as np
from ultralytics import YOLO
from pathlib import Path

PIXELS_PER_MM = 20.6978
THRESHOLD = 120.0

model = YOLO('D:/spring-defects/runs/pose/spring_pose_v6-2/weights/best.pt')
train_dir = Path('D:/spring-defects/data/yolo_pose/train/images')

tp = fp = tn = fn = 0

for img_path in sorted(train_dir.glob('*.jpeg')):
    num = int(img_path.stem.split('_')[1])
    true_label = 'GOOD' if num <= 100 else 'BAD'

    res = model.predict(source=str(img_path), imgsz=960, conf=0.25, device=0, verbose=False)[0]
    if res.keypoints is None or len(res.keypoints.data) == 0:
        continue

    kd = res.keypoints.data.cpu().numpy()[0]
    kpt0, kpt6 = kd[0,:2], kd[6,:2]
    total_mm = float(np.sqrt((kpt0[0]-kpt6[0])**2 + (kpt0[1]-kpt6[1])**2)) / PIXELS_PER_MM
    pred_label = 'BAD' if total_mm >= THRESHOLD else 'GOOD'

    if true_label == 'BAD'  and pred_label == 'BAD':  tp += 1
    if true_label == 'GOOD' and pred_label == 'BAD':  fp += 1
    if true_label == 'GOOD' and pred_label == 'GOOD': tn += 1
    if true_label == 'BAD'  and pred_label == 'GOOD': fn += 1

print(f'Threshold: {THRESHOLD} mm')
print(f'TP (BAD correctly detected)  : {tp}')
print(f'FP (GOOD wrongly flagged)    : {fp}')
print(f'TN (GOOD correctly passed)   : {tn}')
print(f'FN (BAD missed)              : {fn}')
print(f'Accuracy  : {(tp+tn)/(tp+fp+tn+fn)*100:.1f}%')
print(f'Precision : {tp/(tp+fp)*100:.1f}%' if tp+fp > 0 else 'Precision: N/A')
print(f'Recall    : {tp/(tp+fn)*100:.1f}%' if tp+fn > 0 else 'Recall: N/A')
