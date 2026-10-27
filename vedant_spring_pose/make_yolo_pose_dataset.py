
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import random
import yaml

ROOT = Path(r"C:\Users\OM\OneDrive\Desktop\springtop")
OUT = ROOT / "yolo_pose_dataset"
random.seed(42)

# Keep your original four keypoints in this exact order.
XML_FILES = {
    "good_spring": ROOT / "good" / "annotation.xml",
    "bad_spring": ROOT / "bad" / "annotations.xml",
}

def read_annotations(xml_path, kind):
    records = []
    root = ET.parse(xml_path).getroot()

    for image in root.findall("image"):
        name = image.get("name")
        width = int(image.get("width"))
        height = int(image.get("height"))
        points_tag = image.find("points")

        if points_tag is None:
            print("Skipping image without points:", name)
            continue

        points = [
            tuple(map(float, point.split(",")))
            for point in points_tag.get("points", "").split(";")
        ]

        if len(points) != 4:
            print("Skipping image without exactly 4 points:", name)
            continue

        # Find the image in its own class folder.
        image_path = ROOT / ("good" if kind == "good_spring"
                             else "bad") / "images" / name

        if not image_path.exists():
            print("Missing image:", image_path)
            continue

        if any(not (0 <= x <= width and 0 <= y <= height)
               for x, y in points):
            print("Invalid point coordinates:", name)
            continue

        records.append({
            "name": name,
            "path": image_path,
            "width": width,
            "height": height,
            "points": points,
            "class_id": 0 if kind == "good_spring" else 1,
            "kind": kind,
        })

    return records

def estimate_box(points, width, height):
    # Approximate box based on keypoint spread.
    # Review the boxes visually before training.
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    margin_x = width * 0.15
    margin_y = height * 0.05

    x0 = max(0, min(xs) - margin_x)
    y0 = max(0, min(ys) - margin_y)
    x1 = min(width, max(xs) + margin_x)
    y1 = min(height, max(ys) + margin_y)

    # Ensure the box has nonzero size.
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid estimated bounding box")

    return x0, y0, x1, y1

records = []
for kind, xml_path in XML_FILES.items():
    records.extend(read_annotations(xml_path, kind))

if not records:
    raise RuntimeError("No valid annotated images found.")

random.shuffle(records)

# Split by individual images. If multiple views belong to the same
# physical spring, keep those views in the same split instead.
n = len(records)
n_train = int(n * 0.70)
n_val = int(n * 0.20)

splits = {
    "train": records[:n_train],
    "val": records[n_train:n_train + n_val],
    "test": records[n_train + n_val:],
}

for split, items in splits.items():
    for folder in ("images", "labels"):
        (OUT / split / folder).mkdir(parents=True, exist_ok=True)

    for i, record in enumerate(items):
        # Prefix with class to avoid filename collisions.
        stem = Path(record["name"]).stem
        unique_stem = f'{record["kind"]}_{stem}'

        shutil.copy2(
            record["path"],
            OUT / split / "images" / f"{unique_stem}.jpg"
        )

        x0, y0, x1, y1 = estimate_box(
            record["points"], record["width"], record["height"]
        )

        W, H = record["width"], record["height"]
        values = [
            record["class_id"],
            (x0 + x1) / (2 * W),
            (y0 + y1) / (2 * H),
            (x1 - x0) / W,
            (y1 - y0) / H,
        ]

        for x, y in record["points"]:
            values.extend([x / W, y / H, 2])

        label_path = OUT / split / "labels" / f"{unique_stem}.txt"
        label_path.write_text(
            " ".join(
                str(v) if j == 0 else f"{v:.6f}"
                for j, v in enumerate(values)
            ) + "\n",
            encoding="utf-8"
        )

data_yaml = {
    "path": str(OUT).replace("\\", "/"),
    "train": "train/images",
    "val": "val/images",
    "test": "test/images",
    "kpt_shape": [4, 3],
    "flip_idx": [0, 1, 2, 3],
    "names": {
        0: "good_spring",
        1: "bad_spring",
    },
}

with open(OUT / "data.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(data_yaml, f, sort_keys=False)

print("\nDataset created at:", OUT)
print("Total images:", n)
for split, items in splits.items():
    print(f"{split}: {len(items)}")
print("Keypoints: 4")
print("Classes: good_spring, bad_spring")
print("\nIMPORTANT: Bounding boxes are estimated. Inspect them before training.")
