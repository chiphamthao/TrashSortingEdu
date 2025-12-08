# =============================
# TACO → YOLOv8 Segmentation
# Stratified 80/10/10 Split + Instance-Safe Augmentation (x2)
# =============================

from pathlib import Path
import json, shutil, cv2, random
import numpy as np
import pycocotools.mask as mask_util
from collections import defaultdict
from tqdm import tqdm
import albumentations as A
import matplotlib.pyplot as plt

# ================== CONFIG ==================
root = Path("data")
ann_file = root / "annotations.json"
random.seed(42)
AUG_PER_IMAGE = 2  # 2 augmentations -> 3x train dataset

# ================== CREATE DIRS ==================
train_img = root / "images/train"; train_lbl = root / "labels/train"
val_img = root / "images/val"; val_lbl = root / "labels/val"
test_img = root / "images/test"; test_lbl = root / "labels/test"

for d in [train_img, val_img, test_img, train_lbl, val_lbl, test_lbl]:
    d.mkdir(parents=True, exist_ok=True)

# ================== LOAD COCO ==================
with open(ann_file) as f:
    coco = json.load(f)

ann_by_img = defaultdict(list)
for ann in coco["annotations"]:
    ann_by_img[ann["image_id"]].append(ann)

# ================== BINARY MAPPING ==================
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

# ================== STRATIFIED SPLIT ==================
images = coco["images"].copy()
random.shuffle(images)

rec_imgs, trash_imgs = [], []
for img in images:
    img_id = img["id"]
    classes = set(new_label_map[a["category_id"]] for a in ann_by_img[img_id])
    if 0 in classes: rec_imgs.append(img)
    if 1 in classes: trash_imgs.append(img)

def splits(group):
    n = len(group)
    return group[:int(n*0.8)], group[int(n*0.8):int(n*0.9)], group[int(n*0.9):]

train_rec, val_rec, test_rec = splits(rec_imgs)
train_trash, val_trash, test_trash = splits(trash_imgs)

train_ids = {i["id"] for i in train_rec + train_trash}
val_ids = {i["id"] for i in val_rec + val_trash}
test_ids = {i["id"] for i in test_rec + test_trash}

print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

# ================== SAVE ORIGINAL SPLIT ==================
def save_instance(img_id, image, items, folder):
    out_i = folder["img"] / f"{img_id}.jpg"
    out_l = folder["lbl"] / f"{img_id}.txt"
    cv2.imwrite(str(out_i), image)
    h, w = image.shape[:2]
    with open(out_l, "w") as f:
        for cls, polys in items:
            for poly in polys:
                flat = []
                for x, y in poly:
                    flat.extend([x/w, y/h])
                if len(flat) >= 6:
                    f.write(f"{cls} " + " ".join(f"{v:.6f}" for v in flat) + "\n")

def polys_from_ann(ann):
    seg = ann["segmentation"]
    if isinstance(seg, list):
        return [[(poly[i], poly[i+1]) for i in range(0, len(poly), 2)]
                for poly in seg if len(poly) >= 6]
    return []

print("\n📁 Copying images...")
pbar = tqdm(images)
for img in pbar:
    img_id = img["id"]
    image = cv2.imread(str(root / img["file_name"]))
    anns = ann_by_img[img_id]
    items = [(new_label_map[a["category_id"]], polys_from_ann(a)) for a in anns]

    if img_id in train_ids:
        save_instance(img_id, image, items, {"img": train_img, "lbl": train_lbl})
    elif img_id in val_ids:
        save_instance(img_id, image, items, {"img": val_img, "lbl": val_lbl})
    elif img_id in test_ids:
        save_instance(img_id, image, items, {"img": test_img, "lbl": test_lbl})

pbar.close()

# ================== INSTANCE MASK AUGMENTATION ==================
print("\n✨ Applying instance-safe mask augmentation...")

transform = A.Compose([
    A.HorizontalFlip(p=0.6),
    A.RandomBrightnessContrast(p=0.75),
    A.Affine(scale=(0.8, 1.2), translate_percent=0.1,
             rotate=(-25, 25), p=0.75, fit_output=True),
    A.GaussianBlur(p=0.2),
])

aug_count = 0

for img in tqdm(coco["images"], desc="Augmenting"):
    img_id = img["id"]
    if img_id not in train_ids:
        continue

    img_path = train_img / f"{img_id}.jpg"
    image = cv2.imread(str(img_path))
    if image is None:
        continue
    h, w = image.shape[:2]

    anns = ann_by_img[img_id]

    masks_per_obj = []
    for ann in anns:
        cls = new_label_map[ann["category_id"]]
        rle = mask_util.frPyObjects(ann["segmentation"], h, w)
        m = mask_util.decode(rle)
        if m.ndim == 3:
            m = np.any(m, axis=2).astype(np.uint8)
        masks_per_obj.append((cls, m))

    for k in range(AUG_PER_IMAGE):
        trans = transform(image=image, masks=[m for _, m in masks_per_obj])
        aug_img = trans["image"]
        aug_masks = trans["masks"]

        out_i = train_img / f"{img_id}_aug{k}.jpg"
        out_l = train_lbl / f"{img_id}_aug{k}.txt"
        cv2.imwrite(str(out_i), aug_img)

        with open(out_l, "w") as f:
            for (cls, _), m in zip(masks_per_obj, aug_masks):
                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in cnts:
                    if len(cnt) < 3:
                        continue
                    poly = [(int(x), int(y)) for x, y in cnt.squeeze()]
                    flat = []
                    for x, y in poly:
                        flat.extend([x/aug_img.shape[1], y/aug_img.shape[0]])
                    if len(flat) >= 6:
                        f.write(f"{cls} " + " ".join(map(str, flat)) + "\n")

        aug_count += 1

print(f"\n🎉 Augmentation complete! Added {aug_count} samples")

# ================== VISUAL CHECK ==================
print("\n👀 Preview:", out_i.name)
img_vis = cv2.cvtColor(cv2.imread(str(out_i)), cv2.COLOR_BGR2RGB)
plt.figure(figsize=(5,5)); plt.imshow(img_vis); plt.axis("off"); plt.show()

# ================== YAML ==================
with open(root / "data.yaml", "w") as f:
    f.write("path: data\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("test: images/test\n")
    f.write("names:\n  0: recycling\n  1: trash\n")

print("\n🎯 Dataset READY for YOLOv8 training!")
