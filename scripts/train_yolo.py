import os
from pathlib import Path
from ultralytics import YOLO

def main():
    # Set relative paths correctly
    ROOT = Path(__file__).resolve().parent.parent
    model_path = ROOT / "base_models" / "yolov8n-seg.pt"
    data_path = ROOT / "dataset.yaml"

    model = YOLO(str(model_path))

    model.train(
        data=str(data_path),
        epochs=80,
        imgsz=640,
        batch=8,
        workers=2   
    )

if __name__ == "__main__":
    main()
