# src/server/ws_stream.py
import base64, json, asyncio
from fastapi import WebSocket
from typing import Dict

async def ws_send_frame(ws: WebSocket, frame, dets=None):
    import cv2
    _, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    b64 = base64.b64encode(jpg.tobytes()).decode('ascii')
    payload = {"image": b64, "detections": dets or [], "meta": {"ts": asyncio.get_event_loop().time()}}
    await ws.send_text(json.dumps(payload))
