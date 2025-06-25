from ultralytics import YOLO

# Load YOLOv8n model
model = YOLO("yolov8n.pt")

# Fine-tune on custom dataset
model.train(
    data="WeaponDetection.v3/data.yaml",
    epochs=150,
    imgsz=960,
    batch=8,
    device=0,
    pretrained=True  # tetap True agar pakai weight pretrained
)
