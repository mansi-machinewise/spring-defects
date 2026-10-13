import numpy as np
from ultralytics import YOLO

PIXELS_PER_MM = 20.6978

model = YOLO('D:/spring-defects/runs/pose/spring_pose_v6-2/weights/best.pt')

for img_name in ['spring_001.jpeg', 'spring_002.jpeg', 'spring_003.jpeg']:
    img_path = f'D:/spring-defects/data/yolo_pose/train/images/{img_name}'
    res = model.predict(source=img_path, imgsz=960, conf=0.25, device=0, verbose=False)[0]
    kd = res.keypoints.data.cpu().numpy()[0]

    kpt0 = kd[0, :2]
    kpt1 = kd[1, :2]
    kpt5 = kd[5, :2]
    kpt6 = kd[6, :2]

    total_px = float(np.sqrt((kpt0[0]-kpt6[0])**2 + (kpt0[1]-kpt6[1])**2))
    hook_px  = float(np.sqrt((kpt1[0]-kpt5[0])**2 + (kpt1[1]-kpt5[1])**2))

    total_mm = total_px / PIXELS_PER_MM
    hook_mm  = hook_px  / PIXELS_PER_MM

    total_pass = 120.00 <= total_mm <= 120.50
    hook_pass  = 116.00 <= hook_mm  <= 118.00
    overall    = total_pass and hook_pass

    print(f'Image        : {img_name}')
    print(f'Total length : {total_mm:.2f} mm  -> {"PASS" if total_pass else "FAIL"}')
    print(f'Hook length  : {hook_mm:.2f} mm  -> {"PASS" if hook_pass else "FAIL"}')
    print(f'OVERALL      : {"PASS" if overall else "FAIL"}')
    print()
