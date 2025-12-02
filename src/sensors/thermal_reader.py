# src/sensors/thermal_reader.py
import cv2
import numpy as np

class ThermalReader:
    def __init__(self, source=1):
        self.cap = cv2.VideoCapture(source)

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return False, None, None
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        norm = cv2.normalize(gray, None, 0,255, cv2.NORM_MINMAX).astype('uint8')
        return True, gray, norm

    def release(self):
        if self.cap:
            self.cap.release()