import sys
sys.path.insert(0, 'D:/spring-defects/src/measurement')
from measure_spring import measure_spring

for i in range(1, 11):
    img = f'D:/spring-defects/data/yolo_pose/train/images/spring_{i:03d}.jpeg'
    r = measure_spring(img, save_visualization=False)
    print(f'spring_{i:03d}: total={r["measurements"]["total_length"]["mm"]:.2f}mm  hook={r["measurements"]["hook_length"]["mm"]:.2f}mm  -> {r["overall"]}')
