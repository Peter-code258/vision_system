import os
import json
import shutil

def coco2yolo(coco_json_path, images_dir, output_dir):
    """
    将 COCO 数据集转换为 YOLO 格式
    """
    with open(coco_json_path, 'r') as f:
        data = json.load(f)

    os.makedirs(os.path.join(output_dir, "images/train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images/val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels/train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels/val"), exist_ok=True)

    img_id2file = {img['id']: img['file_name'] for img in data['images']}
    cat_id2idx = {cat['id']: idx for idx, cat in enumerate(data['categories'])}

    for ann in data['annotations']:
        img_id = ann['image_id']
        cat_id = ann['category_id']
        bbox = ann['bbox']  # [x,y,width,height]

        # 转 YOLO 格式: x_center y_center w h (归一化)
        img_file = img_id2file[img_id]
        img_path = os.path.join(images_dir, img_file)
        # 这里假设你可以获取图片大小
        # 需要提前读取图片尺寸
        import cv2
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        x, y, bw, bh = bbox
        x_c = (x + bw/2) / w
        y_c = (y + bh/2) / h
        bw /= w
        bh /= h

        label_str = f"{cat_id2idx[cat_id]} {x_c} {y_c} {bw} {bh}\n"
        label_file = os.path.join(output_dir, "labels/train", os.path.splitext(img_file)[0] + ".txt")
        with open(label_file, "a") as f:
            f.write(label_str)

        shutil.copy(img_path, os.path.join(output_dir, "images/train", img_file))

    print("[INFO] COCO -> YOLO conversion done.")
