# src/detectors/ultralytics_infer.py
from ultralytics import YOLO
import numpy as np, time

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt', conf=0.35, iou=0.45, device=None):
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.device = device

    def detect(self, frame):
        results = self.model.predict(source=frame, conf=self.conf, iou=self.iou, device=self.device, verbose=False)
        dets = []
        if not results: return dets
        res = results[0]
        if res.boxes is None: return dets
        for b in res.boxes:
            xyxy = b.xyxy[0].cpu().numpy().astype(int)
            conf = float(b.conf[0].cpu().numpy())
            cls = int(b.cls[0].cpu().numpy())
            label = self.model.names[cls] if cls in self.model.names else str(cls)
            dets.append({'box':tuple(xyxy.tolist()), 'conf':conf, 'class':cls})
        return dets

    # compat with ONNXDetector interface
    def infer(self, frame):
        t0 = time.time()
        dets = self.detect(frame)
        t1 = time.time()
        return dets, (t1-t0)