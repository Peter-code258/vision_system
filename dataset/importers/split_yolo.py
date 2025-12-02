import os
import random
import shutil

def split_yolo_dataset(dataset_dir, val_ratio=0.2):
    images_dir = os.path.join(dataset_dir, "images/train")
    labels_dir = os.path.join(dataset_dir, "labels/train")

    image_files = os.listdir(images_dir)
    val_count = int(len(image_files) * val_ratio)
    val_files = random.sample(image_files, val_count)

    os.makedirs(os.path.join(dataset_dir, "images/val"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "labels/val"), exist_ok=True)

    for img_file in val_files:
        shutil.move(os.path.join(images_dir, img_file), os.path.join(dataset_dir, "images/val", img_file))
        label_file = os.path.splitext(img_file)[0] + ".txt"
        shutil.move(os.path.join(labels_dir, label_file), os.path.join(dataset_dir, "labels/val", label_file))

    print("[INFO] YOLO dataset split done.")
