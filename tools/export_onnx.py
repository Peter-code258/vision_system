#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from ultralytics import YOLO

def export_onnx(weights, output, opset=12, half=False, dynamic=False, imgsz=640):
    print(f"[Export] Loading model: {weights}")
    model = YOLO(weights)

    print("[Export] Exporting to ONNX...")
    model.export(
        format="onnx",
        opset=opset,
        imgsz=imgsz,
        half=half,
        dynamic=dynamic,
        optimize=True,
        simplify=True
    )

    exported = model.exported_model
    if exported and os.path.exists(exported):
        final_path = output
        os.rename(exported, final_path)
        print(f"[Export] ONNX saved to {final_path}")
    else:
        print("[Export] Export failed.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to .pt model")
    parser.add_argument("--output", type=str, default="best.onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", help="Export FP16")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic batch")
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args()

    export_onnx(
        weights=args.weights,
        output=args.output,
        opset=args.opset,
        half=args.half,
        dynamic=args.dynamic,
        imgsz=args.imgsz
    )

if __name__ == "__main__":
    main()
