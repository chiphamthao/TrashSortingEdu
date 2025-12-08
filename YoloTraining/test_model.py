from ultralytics import YOLO
from pathlib import Path
import os

BEST_MODEL = Path("runs/segment/taco_seg_binary3/weights/best.pt")  # update if exp number changed
DATA_CONFIG = Path("data/data.yaml")
NUM_SAVE_IMAGES = 20

print(f"\n🚀 Loading model: {BEST_MODEL}")
model = YOLO(str(BEST_MODEL))

print("\n📊 Evaluating model on TEST set...")
results = model.val(
    data=str(DATA_CONFIG),
    split="test",
    save=True,
    save_txt=True,
    save_conf=True,
    plots=True,
)

print("\n🎯 Evaluation complete!")

print("\n📈 METRICS — TEST SET")
print(f"Box mAP50:      {results.box.map50:.4f}")
print(f"Box mAP50-95:   {results.box.map:.4f}")
print(f"Mask mAP50:     {results.seg.map50:.4f}")
print(f"Mask mAP50-95:  {results.seg.map:.4f}")

print("\n⏱️ Inference Speed:")
for k, v in results.speed.items():
    print(f"{k}: {v:.2f} ms")

print("\n📁 Saved output to runs/segment/val/")

# -----------------------------
# Copy sample predictions
# -----------------------------
pred_dir = Path("runs/segment/val/predicted")
if pred_dir.exists():
    img_files = list(pred_dir.glob("*.jpg"))[:NUM_SAVE_IMAGES]
    out_dir = Path("test_predictions")
    out_dir.mkdir(exist_ok=True)

    for img in img_files:
        os.system(f"cp '{img}' '{out_dir / img.name}'")

    print(f"\n📸 Saved {len(img_files)} prediction preview images → {out_dir.resolve()}")

print("\n🧪 TESTING COMPLETE!")
