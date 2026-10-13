from pathlib import Path
import xml.etree.ElementTree as ET
import cv2

XML_PATH      = Path(r"C:\Users\mansi\Downloads\test_addition\annotations.xml")
IMAGE_DIR     = Path(r"C:\Users\mansi\Downloads\test_addition")
OUTPUT_IMAGES = Path(r"C:\Users\mansi\Downloads\test_new\images")
OUTPUT_LABELS = Path(r"C:\Users\mansi\Downloads\test_new\labels")

KEYPOINT_ORDER = ["pt1","pt2","pt3","pt4","pt5","pt6","pt7"]

OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUTPUT_LABELS.mkdir(parents=True, exist_ok=True)

def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))

tree = ET.parse(XML_PATH)
root = tree.getroot()
images = root.findall('.//image')
print(f'XML images found: {len(images)}')

success = 0
failed = 0

for index, image in enumerate(images, start=1):
    image_name = image.attrib['name']
    xml_width  = int(float(image.attrib['width']))
    xml_height = int(float(image.attrib['height']))
    print(f'[{index}/{len(images)}] {image_name}')

    image_path = IMAGE_DIR / Path(image_name).name
    if not image_path.exists():
        print(f'  ERROR: Image not found')
        failed += 1
        continue

    img = cv2.imread(str(image_path))
    if img is None:
        print(f'  ERROR: Could not read image')
        failed += 1
        continue

    actual_height, actual_width = img.shape[:2]
    if actual_width != xml_width or actual_height != xml_height:
        print(f'  ERROR: Dimension mismatch XML={xml_width}x{xml_height} Image={actual_width}x{actual_height}')
        failed += 1
        continue

    box = image.find('./box[@label=\"spring\"]')
    if box is None:
        print('  ERROR: spring bounding box not found')
        failed += 1
        continue

    xtl = clamp(float(box.attrib['xtl']), 0, xml_width)
    ytl = clamp(float(box.attrib['ytl']), 0, xml_height)
    xbr = clamp(float(box.attrib['xbr']), 0, xml_width)
    ybr = clamp(float(box.attrib['ybr']), 0, xml_height)

    if xbr <= xtl or ybr <= ytl:
        print('  ERROR: Invalid bounding box')
        failed += 1
        continue

    bbox_x_center = ((xtl + xbr) / 2) / xml_width
    bbox_y_center = ((ytl + ybr) / 2) / xml_height
    bbox_width    = (xbr - xtl) / xml_width
    bbox_height   = (ybr - ytl) / xml_height

    skeleton = image.find('./skeleton[@label=\"spring_pose\"]')
    if skeleton is None:
        print('  ERROR: spring_pose skeleton not found')
        failed += 1
        continue

    points = {}
    for point in skeleton.findall('./points'):
        label = point.attrib['label']
        x, y  = map(float, point.attrib['points'].split(','))
        points[label] = (x, y)

    missing = [kp for kp in KEYPOINT_ORDER if kp not in points]
    if missing:
        print(f'  ERROR: Missing keypoints: {missing}')
        failed += 1
        continue

    values = [0, bbox_x_center, bbox_y_center, bbox_width, bbox_height]
    for kp in KEYPOINT_ORDER:
        x, y = points[kp]
        x = clamp(x, 0, xml_width)
        y = clamp(y, 0, xml_height)
        values.extend([x / xml_width, y / xml_height, 2])

    label_path = OUTPUT_LABELS / (Path(image_name).stem + '.txt')
    label_text = ' '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in values)
    label_path.write_text(label_text + '\n', encoding='utf-8')

    cv2.imwrite(str(OUTPUT_IMAGES / Path(image_name).name), img)
    print(f'  OK')
    success += 1

print(f'\nTotal: {len(images)}  Success: {success}  Failed: {failed}')
print(f'Images: {OUTPUT_IMAGES}')
print(f'Labels: {OUTPUT_LABELS}')
