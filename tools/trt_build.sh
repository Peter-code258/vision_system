#!/bin/bash
set -e

ONNX_PATH=$1
ENGINE_PATH=$2
FP16=$3

if [ -z "$ONNX_PATH" ] || [ -z "$ENGINE_PATH" ]; then
    echo "Usage:"
    echo "  ./trt_build.sh model.onnx output.engine [fp16]"
    exit 1
fi

echo "[TensorRT] Building engine from $ONNX_PATH"

FP16_FLAG=""
if [ "$FP16" == "fp16" ]; then
    FP16_FLAG="--fp16"
    echo "[TensorRT] FP16 enabled"
fi

# dynamic shapes example (640)
/usr/src/tensorrt/bin/trtexec \
    --onnx="$ONNX_PATH" \
    --saveEngine="$ENGINE_PATH" \
    --minShapes=input:1x3x640x640 \
    --optShapes=input:4x3x640x640 \
    --maxShapes=input:8x3x640x640 \
    $FP16_FLAG \
    --workspace=4096 \
    --verbose

echo "[TensorRT] Engine saved to $ENGINE_PATH"
