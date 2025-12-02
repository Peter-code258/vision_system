# src/gui/pyqt_main.py
import sys, threading, time
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QSpinBox, QComboBox
from src.gui.qt_video_widget import QtVideoWidget
from src.detectors.onnx_infer import ONNXDetector
from src.utils.logger import log
import cv2

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vision Control Panel (local)")
        self.resize(1200, 800)
        self.video = QtVideoWidget(self)
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.spin_cam = QSpinBox(); self.spin_cam.setValue(0)
        self.combo_backend = QComboBox(); self.combo_backend.addItems(["onnx","ultralytics"])
        self.lbl_fps = QLabel("FPS: 0")
        h = QHBoxLayout()
        h.addWidget(self.btn_start); h.addWidget(self.btn_stop); h.addWidget(self.spin_cam); h.addWidget(self.combo_backend); h.addWidget(self.lbl_fps)
        layout = QVBoxLayout(); layout.addLayout(h); layout.addWidget(self.video)
        self.setLayout(layout)
        self.cap = None; self.detector = None; self.running=False
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)

    def start(self):
        cam = int(self.spin_cam.value())
        self.cap = cv2.VideoCapture(cam)
        backend = self.combo_backend.currentText()
        if backend == "onnx":
            self.detector = ONNXDetector("models/exported/best.onnx", input_size=640)
        else:
            from src.detectors.ultralytics_infer import ObjectDetector
            self.detector = ObjectDetector(model_path="models/exported/best.pt")
        self.running=True
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        prev = 0
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.02); continue
            dets, lat = self.detector.infer(frame)
            for d in dets:
                x1,y1,x2,y2 = d['box']
                cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
            fps = 1.0/(time.time()-prev) if prev>0 else 0.0
            prev = time.time()
            self.lbl_fps.setText(f"FPS: {fps:.1f}")
            self.video.show_frame(frame)
            time.sleep(0.01)

    def stop(self):
        self.running=False
        if self.cap:
            self.cap.release()
            self.cap=None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow(); w.show()
    sys.exit(app.exec_())