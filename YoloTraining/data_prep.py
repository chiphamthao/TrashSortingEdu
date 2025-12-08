# =============================
# TACO → YOLOv8 Segmentation
# Stratified 80/10/10 Split
# =============================

from pathlib import Path
import json, shutil, cv2, random
import numpy as np
import pycocotools.mask as mask_util
from collections import defaultdict
from tqdm import tqdm

# Config
root = Path("data")
ann_file = root / "annotations.json"

random.seed(42)

# Create dirs
train_img = root / "images/train"
val_img = root / "images/val"
test_img = root / "images/test"
train_lbl = root / "labels/train"
val_lbl = root / "labels/val"
test_lbl = root / "labels/test"

dirs = [train_img, val_img, test_img, train_lbl, val_lbl, test_lbl]
for d in dirs:
    d.mkdir(parents=True, exist_ok=True)

# Load COCO
with open(ann_file) as f:
    coco = json.load(f)

# Ann by image
ann_by_img = defaultdict(list)
for ann in coco["annotations"]:
    ann_by_img[ann["image_id"]].append(ann)

# Binary class mapping (unchanged)
new_label_map = {
  0: 0, 1: 1, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0,
  9: 1, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 1,
  17: 0, 18: 1, 19: 1, 20: 1, 21: 1, 22: 1, 23: 0, 24: 1,
  25: 1, 26: 0, 27: 1, 28: 0, 29: 1, 30: 1, 31: 1, 32: 1,
  33: 0, 34: 0, 35: 1, 36: 1, 37: 1, 38: 1, 39: 1, 40: 1,
  41: 1, 42: 1, 43: 0, 44: 0, 45: 1, 46: 1, 47: 0, 48: 1,
  49: 1, 50: 0, 51: 1, 52: 0, 53: 1, 54: 1, 55: 1, 56: 1,
  57: 1, 58: 1, 59: 1,
}

# Helper funcs
def polys_from_ann(ann):
    seg = ann["segmentation"]
    if isinstance(seg, list):
        return [[(poly[i], poly[i+1]) for i in range(0, len(poly), 2)]
                for poly in seg if len(poly) >= 6]
    return []

def write_label(path, cls, polys, w, h):
    flat = []
    for x, y in polys:
        flat.extend([x/w, y/h])
    if len(flat) >= 6:
        path.write(f"{cls} " + " ".join(f"{v:.6f}" for v in flat) + "\n")

# Stratified by class presence
images = coco["images"].copy()
random.shuffle(images)

rec_imgs = []
trash_imgs = []

for img in images:
    img_id = img["id"]
    classes = set(new_label_map[a["category_id"]] for a in ann_by_img[img_id])
    if 0 in classes: rec_imgs.append(img)
    if 1 in classes: trash_imgs.append(img)

def splits(group):
    n = len(group)
    return group[:int(n * 0.8)], group[int(n * 0.8):int(n * 0.9)], group[int(n * 0.9):]

train_rec, val_rec, test_rec = splits(rec_imgs)
train_trash, val_trash, test_trash = splits(trash_imgs)

train_ids = set(i["id"] for i in train_rec + train_trash)
val_ids = set(i["id"] for i in val_rec + val_trash)
test_ids = set(i["id"] for i in test_rec + test_trash)

print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

count_train = 0
count_val = 0
count_test = 0

# Save images
pbar = tqdm(images, desc="Processing images", unit="img")
for img in pbar:
    img_id = img["id"]
    img_path = root / img["file_name"]
    image = cv2.imread(str(img_path))
    h, w = image.shape[:2]

    anns = ann_by_img[img_id]
    items = [(new_label_map[a["category_id"]], polys_from_ann(a)) for a in anns]

    def dump(image, polys, folder):
        out_i = folder["img"] / f"{img_id}.jpg"
        out_l = folder["lbl"] / f"{img_id}.txt"
        cv2.imwrite(str(out_i), image)
        with open(out_l, "w") as f:
            for cls, pset in polys:
                for poly in pset:
                    write_label(f, cls, poly, w, h)

    if img_id in train_ids:
        dump(image, items, {"img": train_img, "lbl": train_lbl})
        count_train += 1
        pbar.set_postfix({"train": count_train, "val": count_val, "test": count_test})

    elif img_id in val_ids:
        dump(image, items, {"img": val_img, "lbl": val_lbl})
        count_val += 1
        pbar.set_postfix({"train": count_train, "val": count_val, "test": count_test})

    elif img_id in test_ids:
        dump(image, items, {"img": test_img, "lbl": test_lbl})
        count_test += 1
        pbar.set_postfix({"train": count_train, "val": count_val, "test": count_test})

pbar.close()
print(f"\nWritten: {count_train} train, {count_val} val, {count_test} test images")

# YAML
yaml = root / "data.yaml"
with open(yaml, "w") as f:
    f.write("path: data\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("test: images/test\n")
    f.write("names:\n  0: recycling\n  1: trash\n")

print("🎉 Done! Stratified split complete.")
