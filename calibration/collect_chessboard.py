# collect_chessboard.py
import cv2
import os
import time

SAVE_DIR = "./calibration/chessboard_images"
os.makedirs(SAVE_DIR, exist_ok=True)

CAM_IDX = 0  # 默认摄像头索引
BOARD_SIZE = (9, 6)  # 内角点数量
DELAY = 1  # 帧采样间隔，秒

cap = cv2.VideoCapture(CAM_IDX)
count = 0

print("[INFO] Press 's' to save frame for calibration, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Camera frame not read.")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret_cb, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, None)

    display = frame.copy()
    if ret_cb:
        cv2.drawChessboardCorners(display, BOARD_SIZE, corners, ret_cb)

    cv2.imshow("Chessboard Capture", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and ret_cb:
        filename = os.path.join(SAVE_DIR, f"chess_{count:03d}.png")
        cv2.imwrite(filename, frame)
        print(f"[INFO] Saved {filename}")
        count += 1
        time.sleep(DELAY)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
