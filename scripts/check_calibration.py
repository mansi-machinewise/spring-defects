import numpy as np, cv2
from ultralytics import YOLO

model = YOLO('D:/spring-defects/runs/pose/spring_pose_v6-2/weights/best.pt')
img_path = 'D:/spring-defects/data/yolo_pose/train/images/spring_001.jpeg'

img = cv2.imread(img_path)
h, w = img.shape[:2]
print(f'Image size: {w}x{h}')

res = model.predict(source=img_path, imgsz=960, conf=0.25, device=0, verbose=False)[0]
kd = res.keypoints.data.cpu().numpy()[0]

kpt0 = kd[0, :2]
kpt6 = kd[6, :2]

total_px = float(np.sqrt((kpt0[0]-kpt6[0])**2 + (kpt0[1]-kpt6[1])**2))
print(f'kpt0 (hook_tip_top)    : ({kpt0[0]:.1f}, {kpt0[1]:.1f})')
print(f'kpt6 (hook_tip_bottom) : ({kpt6[0]:.1f}, {kpt6[1]:.1f})')
print(f'Total pixel distance   : {total_px:.1f} px')
print(f'If total = 120mm, then pixels_per_mm = {total_px/120.0:.4f}')
print(f'If total = 120.25mm, then pixels_per_mm = {total_px/120.25:.4f}')
