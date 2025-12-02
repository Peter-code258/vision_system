# src/gui/qt_video_widget.py
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QImage, QPixmap
import numpy as np

class QtVideoWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScaledContents(True)

    def show_frame(self, frame):
        # frame is BGR
        if frame is None: return
        h,w = frame.shape[:2]
        rgb = frame[:, :, ::-1]
        qimg = QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg))
