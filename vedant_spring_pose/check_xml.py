
import xml.etree.ElementTree as ET

files = [
    r"good\annotation.xml",
    r"bad\annotations.xml"
]

for file in files:
    root = ET.parse(file).getroot()
    total = 0
    invalid = []

    for image in root.findall("image"):
        total += 1
        points = image.find("points")

        if points is None:
            invalid.append((image.get("name"), "No points"))
            continue

        coords = points.get("points", "").split(";")

        if len(coords) != 4:
            invalid.append((image.get("name"), f"{len(coords)} points"))
            continue

        try:
            for coord in coords:
                x, y = map(float, coord.split(","))
                if not (0 <= x <= float(image.get("width"))
                        and 0 <= y <= float(image.get("height"))):
                    invalid.append((image.get("name"), "Coordinate outside image"))
                    break
        except (ValueError, TypeError):
            invalid.append((image.get("name"), "Invalid coordinates"))

    print(f"\nFile: {file}")
    print("Total images:", total)
    print("Invalid annotations:", len(invalid))

    for item in invalid[:10]:
        print(item)
