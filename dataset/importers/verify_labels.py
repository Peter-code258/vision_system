import cv2
import os

def verify_labels(images_dir, labels_dir, classes):
    """
    可视化检查 YOLO 标签是否正确
    """
    img_files = os.listdir(images_dir)
    for img_file in img_files:
        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0]+".txt")
        img = cv2.imread(img_path)

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    cls_id, x_c, y_c, w, h = map(float, parts)
                    h_img, w_img, _ = img.shape
                    xmin = int((x_c - w/2) * w_img)
                    ymin = int((y_c - h/2) * h_img)
                    xmax = int((x_c + w/2) * w_img)
                    ymax = int((y_c + h/2) * h_img)
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
                    cv2.putText(img, classes[int(cls_id)], (xmin, ymin-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        cv2.imshow("Label Verify", img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
