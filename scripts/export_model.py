from ultralytics import YOLO

model = YOLO("runs/segment/train/weights/best.pt")
model.export(format="onnx", imgsz=640, simplify=True)

print("Export completed")

