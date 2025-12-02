# calibrate_camera.py
import cv2
import numpy as np
import glob
import json
import os

IMAGE_DIR = "./calibration/chessboard_images"
OUTPUT_FILE = "./models/homography/camera_params.json"
BOARD_SIZE = (9, 6)  # 内角点数量
SQUARE_SIZE = 0.025  # 实际棋盘方格边长(m)

# 准备世界坐标系点
objp = np.zeros((BOARD_SIZE[0]*BOARD_SIZE[1],3), np.float32)
objp[:,:2] = np.mgrid[0:BOARD_SIZE[0],0:BOARD_SIZE[1]].T.reshape(-1,2)
objp = objp * SQUARE_SIZE

objpoints = []
imgpoints = []

images = glob.glob(os.path.join(IMAGE_DIR, "*.png"))

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, None)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001))
        imgpoints.append(corners2)

if len(objpoints) == 0:
    print("[ERROR] No chessboard detected!")
    exit(1)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 保存参数
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
params = {'mtx': mtx.tolist(), 'dist': dist.tolist()}
with open(OUTPUT_FILE, 'w') as f:
    json.dump(params, f, indent=4)

print(f"[INFO] Camera calibration saved to {OUTPUT_FILE}")
