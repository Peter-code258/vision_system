# src/inference/runner.py
import threading, time
from typing import Callable
import cv2
from src.utils.logger import log

class InferenceRunner:
    def __init__(self, detector, cam_idx=0, thermal_reader=None, fusion_H=None, callback:Callable=None):
        self.detector = detector
        self.cam_idx = cam_idx
        self.cap = None
        self.running = False
        self.thread = None
        self.thermal_reader = thermal_reader
        self.fusion_H = fusion_H
        self.callback = callback  # callback(frame, dets, meta)
        self.last_dets = []

    def start(self):
        self.cap = cv2.VideoCapture(self.cam_idx)
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.02); continue
            dets, lat = self.detector.infer(frame)
            self.last_dets = dets
            # fusion if thermal reader present
            fused = frame
            if self.thermal_reader:
                okt, rawt, normt = self.thermal_reader.read()
                if okt:
                    from src.fusion.thermal_fusion import fuse_rgb_and_thermal, load_homography
                    H = self.fusion_H
                    fused, warped = fuse_rgb_and_thermal(frame, normt, H=H)
            meta = {"latency": lat, "ts": time.time()}
            if self.callback:
                try:
                    self.callback(fused, dets, meta)
                except Exception:
                    log.exception("callback error")
            time.sleep(0.01)

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
