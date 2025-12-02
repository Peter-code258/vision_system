# src/server/main_api.py
import os, time, threading, json
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, JSONResponse
from typing import List
import cv2
from src.utils.logger import log
from src.detectors.onnx_infer import ONNXDetector
from src.detectors.ultralytics_infer import ObjectDetector

app = FastAPI()
# serve static built frontend if exists
dist_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..","web","dist"))
if os.path.exists(dist_dir):
    app.mount("/", StaticFiles(directory=dist_dir, html=True), name="web")

STATE = {
    "backend": "onnx",
    "onnx_path": None,
    "trt_path": None,
    "running": False,
    "cap": None,
    "detector": None,
    "homography": None,
    "last_frame": None,
    "last_dets": []
}
ws_clients: List[WebSocket] = []

@app.post("/upload_model/onnx")
async def upload_onnx(file: UploadFile = File(...)):
    os.makedirs("models/exported", exist_ok=True)
    out = os.path.join("models/exported", file.filename)
    with open(out, "wb") as f:
        f.write(await file.read())
    STATE["onnx_path"] = out
    log.info(f"ONNX uploaded: {out}")
    return {"ok": True, "path": out}

@app.post("/set_backend")
async def set_backend(backend: str = Form(...)):
    STATE["backend"] = backend
    log.info("backend set to %s", backend)
    return {"ok": True, "backend": backend}

@app.get("/status")
def status():
    return {"backend": STATE["backend"], "onnx_path": STATE["onnx_path"], "running": STATE["running"]}

@app.get("/video_feed")
def video_feed():
    if STATE.get("cap") is None:
        STATE["cap"] = cv2.VideoCapture(0)
    return StreamingResponse(_mjpeg_gen(STATE["cap"]), media_type='multipart/x-mixed-replace; boundary=frame')

def _mjpeg_gen(cap):
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02); continue
        _, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.append(ws)
    try:
        while True:
            if STATE.get("last_frame") is None:
                await ws.send_text(json.dumps({"status":"no_frame"}))
                await ws.receive_text()
                continue
            import base64
            _, jpg = cv2.imencode(".jpg", STATE["last_frame"], [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            b64 = base64.b64encode(jpg.tobytes()).decode('ascii')
            payload = {"image": b64, "detections": STATE.get("last_dets", [])}
            await ws.send_text(json.dumps(payload))
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_clients.remove(ws)

@app.post("/start")
def start(backend: str = Form(None)):
    if backend:
        STATE["backend"] = backend
    if STATE["running"]:
        return {"ok": False, "reason": "already running"}
    if STATE["backend"] == "onnx":
        path = STATE.get("onnx_path") or "models/exported/best.onnx"
        STATE["detector"] = ONNXDetector(path, input_size=640)
    elif STATE["backend"] == "ultralytics":
        STATE["detector"] = ObjectDetector(model_path="models/exported/best.pt")
    else:
        return {"ok": False, "reason": "unsupported backend"}
    STATE["cap"] = cv2.VideoCapture(0)
    STATE["running"] = True
    threading.Thread(target=_inference_loop, daemon=True).start()
    log.info("inference started with backend %s", STATE["backend"])
    return {"ok": True}

def _inference_loop():
    det = STATE["detector"]
    cap = STATE["cap"]
    while STATE["running"]:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02); continue
        dets, lat = det.infer(frame)
        STATE["last_frame"] = frame
        STATE["last_dets"] = dets
        # notify WS clients (fire-and-forget)
        for ws in list(ws_clients):
            try:
                import base64, json
                _, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                b64 = base64.b64encode(jpg.tobytes()).decode('ascii')
                payload = {"image": b64, "detections": dets}
                # send in background (no await here)
                import asyncio
                asyncio.run(ws.send_text(json.dumps(payload)))
            except Exception:
                pass
        time.sleep(0.01)

@app.post("/stop")
def stop():
    STATE["running"] = False
    if STATE.get("cap"):
        try: STATE["cap"].release()
        except: pass
        STATE["cap"] = None
    STATE["detector"] = None
    return {"ok": True}

# minimal train/eval triggers (lightweight wrappers)
@app.post("/train")
def train(dataset_yaml: str = Form(...), epochs: int = Form(50)):
    def _worker():
        from src.training.train import train_yolo
        train_yolo(dataset_yaml, epochs=int(epochs))
    threading.Thread(target=_worker, daemon=True).start()
    return {"ok": True, "msg": "training started"}

@app.post("/evaluate")
def evaluate(model_path: str = Form(...)):
    from src.training.evaluate import evaluate_model
    res = evaluate_model(model_path)
    return {"ok": True, "result": res}
