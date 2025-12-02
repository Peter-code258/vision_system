import xml.etree.ElementTree as ET
import os
import shutil
import cv2

def voc2yolo(voc_dir, output_dir, classes):
    """
    VOC XML -> YOLO txt
    """
    os.makedirs(os.path.join(output_dir, "images/train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels/train"), exist_ok=True)

    xml_files = [f for f in os.listdir(voc_dir) if f.endswith(".xml")]

    for xml_file in xml_files:
        tree = ET.parse(os.path.join(voc_dir, xml_file))
        root = tree.getroot()
        img_file = root.find('filename').text
        img_path = os.path.join(voc_dir, img_file)
        img = cv2.imread(img_path)
        h, w, _ = img.shape

        lines = []
        for obj in root.findall('object'):
            cls_name = obj.find('name').text
            if cls_name not in classes:
                continue
            cls_id = classes.index(cls_name)
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            x_c = (xmin + xmax)/2 / w
            y_c = (ymin + ymax)/2 / h
            bw = (xmax - xmin)/w
            bh = (ymax - ymin)/h
            lines.append(f"{cls_id} {x_c} {y_c} {bw} {bh}\n")

        if lines:
            with open(os.path.join(output_dir, "labels/train", os.path.splitext(img_file)[0]+".txt"), "w") as f:
                f.writelines(lines)
            shutil.copy(img_path, os.path.join(output_dir, "images/train", img_file))

    print("[INFO] VOC -> YOLO conversion done.")
