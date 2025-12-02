# src/server/mjpeg_stream.py
import cv2, time
from fastapi.responses import StreamingResponse

def mjpeg_generator(cap, lock=None, quality=80):
    while True:
        if cap is None:
            time.sleep(0.05); continue
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02); continue
        _, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        chunk = jpg.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + chunk + b'\r\n')

def mjpeg_response(cap):
    return StreamingResponse(mjpeg_generator(cap), media_type='multipart/x-mixed-replace; boundary=frame')
