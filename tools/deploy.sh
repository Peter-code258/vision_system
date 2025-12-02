#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PT="$ROOT/models/exported/best.pt"
ONNX="$ROOT/models/exported/best.onnx"
API_HOST="${API_HOST:-http://localhost:8000}"
START_SERVER="${START_SERVER:-0}"  # set to 1 to start uvicorn in this script (useful for dev)

if [ ! -f "$PT" ]; then
  echo "[ERROR] best.pt not found at $PT"
  exit 1
fi

echo "[STEP] Exporting ONNX from $PT -> $ONNX"
python3 "$ROOT/tools/export_onnx.py" --weights "$PT" --output "$ONNX" --imgsz 640 --half || {
  echo "[WARN] export_onnx.py failed (try without --half). Retrying with CPU export..."
  python3 "$ROOT/tools/export_onnx.py" --weights "$PT" --output "$ONNX" --imgsz 640 || exit 1
}

echo "[STEP] Uploading ONNX to backend: $API_HOST/upload_model/onnx"
resp=$(curl -s -F "file=@${ONNX}" "${API_HOST}/upload_model/onnx")
echo "[UPLOAD RESP] $resp"

echo "[STEP] Setting backend to onnx"
curl -s -X POST -F "backend=onnx" "${API_HOST}/set_backend" | jq || true

echo "[STEP] Starting inference loop via API"
curl -s -X POST "${API_HOST}/start" | jq || true

if [ "$START_SERVER" -eq 1 ]; then
  echo "[STEP] Starting uvicorn server (dev) in background..."
  # adjust module path as appropriate
  uvicorn src.server.main_api:app --host 0.0.0.0 --port 8000 --reload &
  echo "[INFO] uvicorn started (pid $!)"
fi

echo "[DONE] deploy.sh completed."
