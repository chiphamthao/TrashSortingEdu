# =============================
# YOLOv8 Segmentation Training Script
# =============================
from ultralytics import YOLO
from pathlib import Path

root = Path("data")
yaml_path = root / "data.yaml"

# Load segmentation model weights
model = YOLO("yolov8m-seg.pt")

# Start training
results = model.train(
    data=str(yaml_path),
    epochs=50,
    imgsz=960,
    batch=8,   # modify based on GPU memory
    save_period=10,
    name='taco_seg_binary'
)

print("🎯 Training complete!")
print(f"Best weights saved in: runs/segment/taco_seg_binary/weights/best.pt")
