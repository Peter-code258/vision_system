#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate_and_align.py

用途：
  - 计算热像 (thermal) -> RGB 的单应性矩阵 (homography)
  - 支持方法：chessboard（自动角点）、feature（ORB+RANSAC）、manual（交互点击）
  - 将结果保存为 models/homography/homography.json（包含 H、meta）
  - 可保存/加载配对点（.npy）

用法示例：
  python calibrate_and_align.py --rgb_dir ./calibration/rgb_images --thermal_dir ./calibration/thermal_images --method feature
  python calibrate_and_align.py --rgb_dir ./calibration/rgb_images --thermal_dir ./calibration/thermal_images --method manual
"""

import os
import cv2
import numpy as np
import json
import argparse
from datetime import datetime
from src.utils.camera_calib_io import load_homography
H = load_homography("models/homography/homography.json")

def list_image_files(d):
    exts = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    files = []
    for f in sorted(os.listdir(d)):
        if os.path.splitext(f)[1].lower() in exts:
            files.append(os.path.join(d, f))
    return files

def pair_by_basename(rgb_files, th_files):
    # map stem -> path
    from pathlib import Path
    rgb_map = {Path(p).stem: p for p in rgb_files}
    th_map  = {Path(p).stem: p for p in th_files}
    common = sorted(set(rgb_map.keys()) & set(th_map.keys()))
    pairs = [(rgb_map[k], th_map[k]) for k in common]
    if pairs:
        return pairs
    # fallback: pair by index
    n = min(len(rgb_files), len(th_files))
    return list(zip(rgb_files[:n], th_files[:n]))

def detect_chessboard_corners(img, board=(9,6)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
    ret, corners = cv2.findChessboardCorners(gray, board, None)
    if not ret:
        return None
    corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001))
    pts = corners2.reshape(-1,2)
    return pts

def auto_chessboard_homography(rgb_path, th_path, board=(9,6)):
    rgb = cv2.imread(rgb_path)
    th  = cv2.imread(th_path)
    rgb_pts = detect_chessboard_corners(rgb, board)
    th_pts  = detect_chessboard_corners(th, board)
    if rgb_pts is None or th_pts is None:
        raise RuntimeError("Chessboard corners not found in one of the images.")
    # need same ordering: findChessboardCorners returns consistent ordering for both views
    H, mask = cv2.findHomography(th_pts, rgb_pts, cv2.RANSAC, 5.0)
    return H, rgb_pts, th_pts

def feature_match_homography(rgb_path, th_path, max_kp=2000):
    img1 = cv2.imread(th_path, cv2.IMREAD_GRAYSCALE)  # thermal as source
    img2 = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE) # rgb as dest
    if img1 is None or img2 is None:
        raise RuntimeError("Unable to read input images for feature matching.")
    # ORB detector
    orb = cv2.ORB_create(nfeatures=1500)
    k1, d1 = orb.detectAndCompute(img1, None)
    k2, d2 = orb.detectAndCompute(img2, None)
    if d1 is None or d2 is None:
        raise RuntimeError("No features found by ORB.")
    # matcher
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(d1, d2)
    matches = sorted(matches, key=lambda x: x.distance)
    if len(matches) < 8:
        raise RuntimeError(f"Not enough matches ({len(matches)}) to compute homography.")
    # take top matches
    best = matches[:min(len(matches), 200)]
    pts1 = np.float32([k1[m.queryIdx].pt for m in best])
    pts2 = np.float32([k2[m.trainIdx].pt for m in best])
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    return H, pts2, pts1, best, k1, k2, mask

def manual_point_collection(rgb_path, th_path, save_pairs_dir=None):
    """
    交互式：在两个窗口分别点击对应点（先在 RGB 窗口点击点，再在 thermal 窗口点击对应点）
    键位： 
      - 'c' 计算 homography（需要 >=4 对点）
      - 'r' 重置当前对（删除最近一对）
      - 's' 保存点对到 npy（如果给定了 save_pairs_dir）
      - 'q' 退出（不会保存除非调用保存）
    """
    rgb = cv2.imread(rgb_path)
    th  = cv2.imread(th_path)
    if rgb is None or th is None:
        raise RuntimeError("读取图片失败，请检查路径")

    rgb_pts = []
    th_pts = []
    current_mode = "rgb"  # expect click on rgb first then thermal

    win_rgb = "RGB"
    win_th  = "THERMAL"
    rgb_disp = rgb.copy()
    th_disp  = th.copy()

    def on_rgb_click(event, x, y, flags, param):
        nonlocal rgb_disp, current_mode
        if event == cv2.EVENT_LBUTTONDOWN:
            rgb_pts.append((x,y))
            cv2.circle(rgb_disp, (x,y), 4, (0,255,0), -1)
            cv2.putText(rgb_disp, f"{len(rgb_pts)}", (x+6,y+6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    def on_th_click(event, x, y, flags, param):
        nonlocal th_disp, current_mode
        if event == cv2.EVENT_LBUTTONDOWN:
            th_pts.append((x,y))
            cv2.circle(th_disp, (x,y), 4, (0,255,0), -1)
            cv2.putText(th_disp, f"{len(th_pts)}", (x+6,y+6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.namedWindow(win_rgb, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_th, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_rgb, on_rgb_click)
    cv2.setMouseCallback(win_th, on_th_click)

    instructions = "[Manual Mode] Click matching points: first click RGB then corresponding THERMAL. Keys: c=compute, r=undo last pair, s=save pts, q=quit"
    print(instructions)

    while True:
        cv2.imshow(win_rgb, rgb_disp)
        cv2.imshow(win_th, th_disp)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            print("退出，不保存。")
            break
        elif key == ord('r'):
            if rgb_pts:
                rgb_pts.pop(); th_pts.pop()
                rgb_disp = rgb.copy(); th_disp = th.copy()
                for i, p in enumerate(rgb_pts):
                    cv2.circle(rgb_disp, (int(p[0]), int(p[1])), 4, (0,255,0), -1)
                    cv2.putText(rgb_disp, f"{i+1}", (int(p[0])+6,int(p[1])+6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                for i, p in enumerate(th_pts):
                    cv2.circle(th_disp, (int(p[0]), int(p[1])), 4, (0,255,0), -1)
                    cv2.putText(th_disp, f"{i+1}", (int(p[0])+6,int(p[1])+6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                print("撤销最后一对点。")
        elif key == ord('s'):
            if save_pairs_dir:
                os.makedirs(save_pairs_dir, exist_ok=True)
                base = os.path.splitext(os.path.basename(rgb_path))[0]
                np.save(os.path.join(save_pairs_dir, f"{base}_rgb_pts.npy"), np.array(rgb_pts))
                np.save(os.path.join(save_pairs_dir, f"{base}_th_pts.npy"), np.array(th_pts))
                print("已保存点对到:", save_pairs_dir)
            else:
                print("未指定保存目录，跳过保存。")
        elif key == ord('c'):
            if len(rgb_pts) >= 4 and len(th_pts) >= 4 and len(rgb_pts) == len(th_pts):
                pts_rgb = np.array(rgb_pts, dtype=np.float32)
                pts_th  = np.array(th_pts, dtype=np.float32)
                H, mask = cv2.findHomography(pts_th, pts_rgb, cv2.RANSAC, 5.0)
                cv2.destroyAllWindows()
                return H, pts_rgb, pts_th
            else:
                print("至少需要 4 对点且两侧点数相同。当前：", len(rgb_pts), len(th_pts))
    cv2.destroyAllWindows()
    return None, None, None

def save_homography(H, out_path, meta):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    data = {"H": H.tolist(), "meta": meta}
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print("Saved homography to:", out_path)

def visualize_and_save(rgb_path, th_path, H, out_vis=None):
    rgb = cv2.imread(rgb_path)
    th  = cv2.imread(th_path)
    if H is None:
        return
    warped = cv2.warpPerspective(th, H, (rgb.shape[1], rgb.shape[0]))
    heat = warped.copy()
    # pseudo color for thermal-like visualization
    if heat.ndim == 3 and heat.shape[2] == 3:
        gray = cv2.cvtColor(heat, cv2.COLOR_BGR2GRAY)
    else:
        gray = heat if heat.ndim==2 else cv2.cvtColor(heat, cv2.COLOR_BGR2GRAY)
    heat_color = cv2.applyColorMap(cv2.normalize(gray, None, 0,255, cv2.NORM_MINMAX).astype('uint8'), cv2.COLORMAP_JET)
    fused = cv2.addWeighted(rgb, 0.6, heat_color, 0.4, 0)
    if out_vis:
        os.makedirs(os.path.dirname(out_vis), exist_ok=True)
        cv2.imwrite(out_vis, fused)
        print("Saved visualization to:", out_vis)
    # show briefly
    cv2.imshow("Fused preview", fused)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

def main(args):
    rgb_dir = args.rgb_dir
    th_dir = args.thermal_dir
    method = args.method.lower()
    out = args.out
    board = tuple(map(int, args.board.split('x'))) if args.board else (9,6)
    save_pairs_dir = args.save_pairs

    if not os.path.isdir(rgb_dir):
        raise SystemExit("rgb_dir not found: " + rgb_dir)
    if not os.path.isdir(th_dir):
        raise SystemExit("thermal_dir not found: " + th_dir)

    rgb_files = list_image_files(rgb_dir)
    th_files  = list_image_files(th_dir)
    if not rgb_files or not th_files:
        raise SystemExit("未找到图片，请检查目录（需要至少一张 RGB 与一张 Thermal）")

    pairs = pair_by_basename(rgb_files, th_files)
    if not pairs:
        raise SystemExit("未找到配对图像，请检查文件名或使用索引配对")

    # pick the first pair by default (可扩展为多对平均)
    rgb_path, th_path = pairs[0]
    print("使用配对：\n  RGB:", rgb_path, "\n  TH:", th_path)
    H = None; pts_rgb = None; pts_th = None; method_used = None

    try:
        if method == "chessboard":
            print("尝试通过棋盘格角点自动配准，board=", board)
            H, pts_rgb, pts_th = auto_chessboard_homography(rgb_path, th_path, board=board)
            method_used = "chessboard"
        elif method == "feature":
            print("尝试通过特征点 (ORB) 自动配准...")
            H, pts_rgb, pts_th, matches, k1, k2, mask = feature_match_homography(rgb_path, th_path)
            method_used = "feature"
        elif method == "manual":
            print("进入交互模式，请在窗口上点击对应点...")
            H, pts_rgb, pts_th = manual_point_collection(rgb_path, th_path, save_pairs_dir=save_pairs_dir)
            method_used = "manual"
        elif method == "auto":
            # try chessboard first, fallback to feature, then manual
            try:
                H, pts_rgb, pts_th = auto_chessboard_homography(rgb_path, th_path, board=board)
                method_used = "chessboard"
                print("自动选择：chessboard 成功")
            except Exception as e1:
                print("chessboard 失败:", e1, "尝试 feature...")
                try:
                    H, pts_rgb, pts_th, *_ = feature_match_homography(rgb_path, th_path)
                    method_used = "feature"
                    print("自动选择：feature 成功")
                except Exception as e2:
                    print("feature 失败:", e2)
                    print("降级到 manual 交互模式")
                    H, pts_rgb, pts_th = manual_point_collection(rgb_path, th_path, save_pairs_dir=save_pairs_dir)
                    method_used = "manual"
        else:
            raise SystemExit("Unknown method: " + method)
    except Exception as e:
        raise SystemExit("配准失败: " + str(e))

    if H is None:
        raise SystemExit("未能计算出单应矩阵 (H)")

    # save homography
    meta = {
        "method": method_used,
        "rgb_image": os.path.abspath(rgb_path),
        "thermal_image": os.path.abspath(th_path),
        "rgb_shape": None,
        "thermal_shape": None,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    try:
        import imageio
        rgb_tmp = imageio.imread(rgb_path)
        th_tmp = imageio.imread(th_path)
        meta["rgb_shape"] = list(rgb_tmp.shape)
        meta["thermal_shape"] = list(th_tmp.shape)
    except Exception:
        # fallback via cv2
        rgb_tmp = cv2.imread(rgb_path); th_tmp = cv2.imread(th_path)
        if rgb_tmp is not None: meta["rgb_shape"] = list(rgb_tmp.shape)
        if th_tmp is not None: meta["thermal_shape"] = list(th_tmp.shape)

    save_homography(H, out, meta)

    # optional visualize and save
    if args.visualize:
        vis_out = args.visualize if isinstance(args.visualize, str) else os.path.join(os.path.dirname(out),"homography_preview.jpg")
        visualize_and_save(rgb_path, th_path, H, out_vis=vis_out)

    print("Done. 方法:", method_used)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Calibrate and align thermal -> RGB (compute homography)")
    p.add_argument("--rgb_dir", type=str, required=True, help="RGB images directory (paired)")
    p.add_argument("--thermal_dir", type=str, required=True, help="Thermal images directory (paired)")
    p.add_argument("--out", type=str, default="./models/homography/homography.json", help="Output homography JSON")
    p.add_argument("--method", type=str, default="auto", choices=["auto","chessboard","feature","manual"], help="alignment method")
    p.add_argument("--board", type=str, default="9x6", help="chessboard internal corners (WxH), e.g. 9x6")
    p.add_argument("--save_pairs", type=str, default="./calibration/pts/", help="保存手动点对的目录（可选）")
    p.add_argument("--visualize", type=str, nargs='?', const=True, help="保存并显示融合预览图（可选：提供输出路径）")
    args = p.parse_args()
    main(args)
