#!/usr/bin/env python3
"""
COCO → YOLO Converter (Enhanced Version)
Author: vision_system project

Features:
- Auto split train/val/test
- High-speed multithreaded image copy
- BBox validation
- COCO categories → YOLO names
- Auto-generate dataset.yaml
- Optional class filtering
- Class distribution statistics
- tqdm progress bars
"""

import json
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import random
import os


def load_coco(coco_json):
    with open(coco_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def xywh_to_yolo(x, y, w, h, img_w, img_h):
    """Convert COCO bbox (x,y,w,h) to YOLO (cx,cy,w,h) normalized."""
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    return cx, cy, w / img_w, h / img_h


def validate_bbox(x, y, w, h, img_w, img_h):
    """Check if bbox is valid."""
    if w <= 0 or h <= 0:
        return False
    if x < 0 or y < 0:
        return False
    if x + w > img_w or y + h > img_h:
        return False
    return True


def copy_image(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_label_file(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for line in lines:
            f.write(line + "\n")


def distribute_dataset(images_list, split):
    """Split into train/val/test"""
    random.shuffle(images_list)
    n = len(images_list)
    t1 = int(split[0] * n)
    t2 = int((split[0] + split[1]) * n)
    train = images_list[:t1]
    val = images_list[t1:t2]
    test = images_list[t2:] if split[2] > 0 else []
    return train, val, test


def save_dataset_yaml(out, nc, names):
    yaml_path = out / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"train: {out/'images/train'}\n")
        f.write(f"val: {out/'images/val'}\n")
        if (out / "images/test").exists():
            f.write(f"test: {out/'images/test'}\n")
        f.write("\n")
        f.write(f"nc: {nc}\n")
        f.write("names:\n")
        for i, name in names.items():
            f.write(f"  {i}: {name}\n")

    print(f"\nGenerated dataset.yaml → {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert COCO dataset to YOLO format")
    parser.add_argument("--coco", type=str, required=True, help="Path to COCO annotations (json)")
    parser.add_argument("--images", type=str, required=True, help="Path to COCO images folder")
    parser.add_argument("--out", type=str, required=True, help="Output YOLO directory")
    parser.add_argument("--split", nargs=3, type=float, default=[0.8, 0.1, 0.1], help="Train/Val/Test split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--keep-classes", nargs="+", type=str, help="Only keep specific categories (names)")
    args = parser.parse_args()

    random.seed(args.seed)
    out = Path(args.out)

    # Load COCO data
    coco = load_coco(args.coco)
    images = {img["id"]: img for img in coco["images"]}
    annotations = coco["annotations"]

    # Category map
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}

    # Build YOLO class map
    if args.keep_classes:
        keep = set(args.keep_classes)
        selected = {cid: name for cid, name in cat_id_to_name.items() if name in keep}
        print(f"Filtering classes: {selected}")
    else:
        selected = cat_id_to_name

    name_to_new_id = {name: i for i, name in enumerate(selected.values())}
    cat_id_to_new = {cid: name_to_new_id[name] for cid, name in selected.items()}

    # Group annotations by image
    image_to_anns = {}
    for ann in annotations:
        cid = ann["category_id"]
        if cid not in selected:
            continue
        img_id = ann["image_id"]
        if img_id not in image_to_anns:
            image_to_anns[img_id] = []
        image_to_anns[img_id].append(ann)

    valid_images = []
    labels_output = {}

    print("Processing annotations...")
    for img_id, anns in tqdm(image_to_anns.items(), desc="Formatting labels"):
        img = images[img_id]
        img_w, img_h = img["width"], img["height"]
        filename = img["file_name"]
        img_path = Path(args.images) / filename

        if not img_path.exists():
            continue

        lines = []
        valid = True

        for ann in anns:
            x, y, w, h = ann["bbox"]
            if not validate_bbox(x, y, w, h, img_w, img_h):
                valid = False
                break
            cx, cy, nw, nh = xywh_to_yolo(x, y, w, h, img_w, img_h)
            cls_id = cat_id_to_new[ann["category_id"]]
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if valid and lines:
            labels_output[img_path] = lines
            valid_images.append(img_path)

    print(f"Valid images: {len(valid_images)}")

    # Split dataset
    train, val, test = distribute_dataset(valid_images, args.split)

    # Prepare output folders
    for sub in ["images/train", "images/val", "images/test",
                "labels/train", "labels/val", "labels/test"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    # Parallel copy
    def process(img_path, target):
        dst_img = out / "images" / target / img_path.name
        copy_image(img_path, dst_img)
        dst_lbl = out / "labels" / target / (img_path.stem + ".txt")
        write_label_file(dst_lbl, labels_output[img_path])

    print("Copying images (multi-thread)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        for img in train:
            futures.append(executor.submit(process, img, "train"))
        for img in val:
            futures.append(executor.submit(process, img, "val"))
        for img in test:
            futures.append(executor.submit(process, img, "test"))

        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass

    # Save dataset.yaml
    save_dataset_yaml(out, len(selected), {i: name for i, name in enumerate(selected.values())})

    # Stats
    print("\nClass distribution:")
    class_count = {name: 0 for name in selected.values()}

    for lines in labels_output.values():
        for l in lines:
            cls = int(l.split()[0])
            name = list(selected.values())[cls]
            class_count[name] += 1

    for name, cnt in class_count.items():
        print(f"{name:20s}: {cnt}")

    print("\nConversion complete!")


if __name__ == "__main__":
    main()
