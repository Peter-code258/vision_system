# src/fusion/thermal_fusion.py
import cv2, numpy as np, json

def load_homography(path):
    if path.endswith(".npy"):
        return np.load(path)
    with open(path, "r") as f:
        data = json.load(f)
    return np.array(data.get("H"))

def warp_thermal_to_rgb(thermal_img, H, out_shape):
    h, w = out_shape[:2]
    warped = cv2.warpPerspective(thermal_img, H, (w, h), flags=cv2.INTER_LINEAR)
    warped = cv2.normalize(warped, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    return warped

def pseudo_color(heat_uint8, cmap=cv2.COLORMAP_JET):
    return cv2.applyColorMap(heat_uint8, cmap)

def overlay_heatmap(rgb_bgr, heat_bgr, alpha=0.45):
    heat = cv2.resize(heat_bgr, (rgb_bgr.shape[1], rgb_bgr.shape[0]))
    return cv2.addWeighted(rgb_bgr, 1-alpha, heat, alpha, 0)

def roi_stats_from_warped(warped_heatmap, box):
    x1,y1,x2,y2 = box
    h = warped_heatmap[y1:y2, x1:x2]
    if h.size == 0:
        return None
    return float(h.mean()), int(h.max()), float(h.std())

def fuse_rgb_and_thermal(rgb_bgr, thermal_img, H=None, alpha=0.45, colormap=cv2.COLORMAP_JET):
    if H is not None:
        warped = warp_thermal_to_rgb(thermal_img, H, rgb_bgr.shape)
    else:
        warped = cv2.resize(thermal_img, (rgb_bgr.shape[1], rgb_bgr.shape[0]))
        warped = cv2.normalize(warped, None, 0,255, cv2.NORM_MINMAX).astype('uint8')
    heat_color = pseudo_color(warped, cmap=colormap)
    fused = overlay_heatmap(rgb_bgr, heat_color, alpha=alpha)
    return fused, warped