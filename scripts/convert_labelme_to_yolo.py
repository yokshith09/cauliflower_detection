import json
import os
from pathlib import Path

def convert_json_to_yolov8_seg(json_path, output_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    img_width = data["imageWidth"]
    img_height = data["imageHeight"]

    yolo_lines = []

    for shape in data["shapes"]:
        if shape["label"] != "cauliflower":
            continue

        points = shape["points"]

        # Normalize polygon points
        normalized_points = []
        for x, y in points:
            x_norm = x / img_width
            y_norm = y / img_height
            normalized_points.append(f"{x_norm} {y_norm}")

        line = "0 " + " ".join(normalized_points)
        yolo_lines.append(line)

    if yolo_lines:
        txt_path = output_dir / (json_path.stem + ".txt")
        with open(txt_path, "w") as f:
            f.write("\n".join(yolo_lines))


def process_split(split):
    img_dir = Path("images") / split
    lbl_dir = Path("labels") / split

    lbl_dir.mkdir(parents=True, exist_ok=True)

    for json_file in img_dir.glob("*.json"):
        convert_json_to_yolov8_seg(json_file, lbl_dir)


for split in ["train", "val"]:
    process_split(split)

print("Conversion to YOLOv8-seg format completed.")
