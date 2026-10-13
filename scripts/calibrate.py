import cv2
import numpy as np

img_path = r'C:\Users\mansi\Downloads\IMG_20261002_104256.jpg'
known_mm = 100.0  # distance between the two points you will click in mm (e.g. 1cm to 11cm = 100mm)

img = cv2.imread(img_path)
h, w = img.shape[:2]
scale = 800/max(h,w)
display = cv2.resize(img, (int(w*scale), int(h*scale)))

points = []

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
        real_x, real_y = x/scale, y/scale
        points.append((real_x, real_y))
        cv2.circle(display, (x,y), 8, (0,0,255), -1)
        cv2.putText(display, str(len(points)), (x+10,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Calibration', display)
        print(f'Point {len(points)}: ({real_x:.1f}, {real_y:.1f})')

cv2.imshow('Calibration', display)
cv2.setMouseCallback('Calibration', click)
print('Click point 1 (e.g. 1cm mark) then click point 2 (e.g. 11cm mark)')
print('Press Q when done')
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(points) == 2:
    dist_px = np.sqrt((points[1][0]-points[0][0])**2 + (points[1][1]-points[0][1])**2)
    px_per_mm = dist_px / known_mm
    print(f'Pixel distance : {dist_px:.2f} px')
    print(f'Known distance : {known_mm} mm')
    print(f'Pixels per mm  : {px_per_mm:.4f}')
    print(f'mm per pixel   : {1/px_per_mm:.6f}')
else:
    print(f'Only {len(points)} point(s) selected. Need 2.')
