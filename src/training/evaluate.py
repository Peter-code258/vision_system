# src/training/evaluate.py
from ultralytics import YOLO

def evaluate_model(model_path="./models/exported/best.pt"):
    model = YOLO(model_path)
    res = model.val()
    return res