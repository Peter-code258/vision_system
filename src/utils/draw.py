# src/utils/draw.py
import cv2

def draw_detections(img, dets, class_names=None, color=(0,255,0)):
    for d in dets:
        x1,y1,x2,y2 = d['box']
        conf = d.get('conf',0.0)
        cls = d.get('class',0)
        label = f"{class_names[cls] if class_names and cls < len(class_names) else cls} {conf:.2f}"
        cv2.rectangle(img, (x1,y1),(x2,y2), color, 2)
        cv2.putText(img, label, (x1, max(12,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    return img

def draw_ir_status(img, ir_latest):
    x0,y0=10,30
    for i,(k,v) in enumerate(ir_latest.items()):
        cv2.putText(img, f"{k}:{v}", (x0, y0 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    return img
