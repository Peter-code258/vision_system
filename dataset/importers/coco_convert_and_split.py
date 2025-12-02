#!/usr/bin/env python3
# dataset/importers/coco_convert_and_split.py

"""
# 在项目根目录执行
python3 vision_system/dataset/importers/coco_convert_and_split.py \
  --coco /path/to/annotations/instances_train2017.json \
  --images /path/to/images/train2017 \
  --out vision_system/dataset/yolo \
  --split 0.8,0.1,0.1 \
  --seed 42
"""

import os, json, argparse, shutil, random
from pathlib import Path
from tqdm import tqdm
import cv2

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def coco_to_yolo(coco_json, images_dir, out_dir, split=(0.8, 0.1, 0.1), seed=42, classes=None):
    """
    coco_json: path to COCO annotations json
    images_dir: path to folder that contains COCO images (file names in JSON)
    out_dir: target dataset root (will create images/train,val,test and labels/train,val,test)
    split: tuple (train_ratio, val_ratio, test_ratio) sum must be 1.0
    classes: optional list of category names to keep (None = keep all)
    """
    assert abs(sum(split) - 1.0) < 1e-6, "split must sum to 1.0"
    ensure_dir(out_dir)
    imgs_out = Path(out_dir) / "images"
    labels_out = Path(out_dir) / "labels"
    for s in ["train","val","test"]:
        ensure_dir(imgs_out / s)
        ensure_dir(labels_out / s)

    data = json.load(open(coco_json, "r", encoding="utf-8"))
    # build maps
    id2img = {im["id"]: im for im in data["images"]}
    cats = data.get("categories", [])
    if classes:
        # filter categories, build mapping cat_id -> new idx (0..nc-1)
        keep = [c for c in cats if c["name"] in classes]
        cat_map = {c["id"]: i for i,c in enumerate(keep)}
        names = [c["name"] for c in keep]
    else:
        cats_sorted = sorted(cats, key=lambda x: x["id"])
        cat_map = {c["id"]: i for i,c in enumerate(cats_sorted)}
        names = [c["name"] for c in cats_sorted]

    # accumulate annotations per image
    anns_per_image = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        anns_per_image.setdefault(img_id, []).append(ann)

    # create a list of image entries that have at least one annotation
    img_entries = []
    for img_id, imginfo in id2img.items():
        if img_id in anns_per_image:
            img_entries.append(imginfo)

    # shuffle and split
    random.seed(seed)
    random.shuffle(img_entries)
    n = len(img_entries)
    n_train = int(n * split[0])
    n_val = int(n * split[1])
    train_set = img_entries[:n_train]
    val_set = img_entries[n_train:n_train+n_val]
    test_set = img_entries[n_train+n_val:]

    def process_subset(subset, name):
        for imginfo in tqdm(subset, desc=f"Writing {name}"):
            fname = imginfo["file_name"]
            src = Path(images_dir) / fname
            if not src.exists():
                # try same folder as coco json (common case)
                alt = Path(coco_json).parent / fname
                if alt.exists():
                    src = alt
                else:
                    print(f"[WARN] image not found: {fname}, skipping")
                    continue
            # copy image
            dst_img = imgs_out / name / fname
            shutil.copy2(src, dst_img)
            # write label
            w, h = imginfo.get("width"), imginfo.get("height")
            if not w or not h:
                img = cv2.imread(str(src))
                if img is None:
                    print(f"[WARN] Cannot read image {src}, skipping labels")
                    continue
                h, w = img.shape[:2]
            label_lines = []
            for ann in anns_per_image.get(imginfo["id"], []):
                cid = ann["category_id"]
                if cid not in cat_map:
                    continue
                bbox = ann["bbox"]  # x,y,w,h (absolute)
                x, y, bw, bh = bbox
                x_center = (x + bw / 2.0) / w
                y_center = (y + bh / 2.0) / h
                bw_n = bw / w
                bh_n = bh / h
                label_lines.append(f"{cat_map[cid]} {x_center:.6f} {y_center:.6f} {bw_n:.6f} {bh_n:.6f}")
            # save label file (same basename)
            label_path = labels_out / name / (Path(fname).stem + ".txt")
            with open(label_path, "w", encoding="utf-8") as f:
                f.write("\n".join(label_lines))

    process_subset(train_set, "train")
    process_subset(val_set, "val")
    process_subset(test_set, "test")

    # write dataset yaml file for ultralytics
    out_yaml = Path("vision_system/configs/dataset.yaml")
    dataset_yaml = {
        "train": str((Path(out_dir) / "images" / "train").resolve()),
        "val": str((Path(out_dir) / "images" / "val").resolve()),
        "test": str((Path(out_dir) / "images" / "test").resolve()),
        "nc": len(names),
        "names": names
    }
    with open(out_yaml, "w", encoding="utf-8") as f:
        import yaml
        yaml.safe_dump(dataset_yaml, f)
    print("[DONE] dataset prepared. dataset.yaml written to:", out_yaml)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coco", required=True, help="path to COCO json annotations")
    p.add_argument("--images", required=True, help="path to folder containing COCO images")
    p.add_argument("--out", default="dataset/yolo", help="output dataset root")
    p.add_argument("--split", default="0.8,0.1,0.1", help="train,val,test ratios (comma sep)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--classes", nargs="*", default=None, help="optional list of class names to keep")
    args = p.parse_args()

    split = tuple(map(float, args.split.split(",")))
    if abs(sum(split) - 1.0) > 1e-6:
        raise SystemExit("split must sum to 1.0")

    coco_to_yolo(args.coco, args.images, args.out, split=split, seed=args.seed, classes=args.classes)

if __name__ == "__main__":
    main()
