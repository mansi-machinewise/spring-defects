import numpy as np
from ultralytics import YOLO

PIXELS_PER_MM = 20.6978

# tolerance with small rounding buffer
TOTAL_MIN, TOTAL_MAX = 119.50, 121.00
HOOK_MIN,  HOOK_MAX  = 115.50, 118.50

model = YOLO('D:/spring-defects/runs/pose/spring_pose_v6-2/weights/best.pt')

results = []
for i in range(1, 11):
    img_path = f'D:/spring-defects/data/yolo_pose/train/images/spring_{i:03d}.jpeg'
    try:
        res = model.predict(source=img_path, imgsz=960, conf=0.25, device=0, verbose=False)[0]
        kd = res.keypoints.data.cpu().numpy()[0]
        kpt0, kpt1, kpt5, kpt6 = kd[0,:2], kd[1,:2], kd[5,:2], kd[6,:2]
        total_mm = float(np.sqrt((kpt0[0]-kpt6[0])**2 + (kpt0[1]-kpt6[1])**2)) / PIXELS_PER_MM
        hook_mm  = float(np.sqrt((kpt1[0]-kpt5[0])**2 + (kpt1[1]-kpt5[1])**2)) / PIXELS_PER_MM
        results.append((f'spring_{i:03d}', total_mm, hook_mm))
        print(f'spring_{i:03d}: total={total_mm:.2f}mm  hook={hook_mm:.2f}mm')
    except:
        print(f'spring_{i:03d}: skipped')

totals = [r[1] for r in results]
hooks  = [r[2] for r in results]
print(f'\nTotal length: min={min(totals):.2f}  max={max(totals):.2f}  mean={np.mean(totals):.2f}')
print(f'Hook length : min={min(hooks):.2f}  max={max(hooks):.2f}  mean={np.mean(hooks):.2f}')
