# src/utils/camera_calib_io.py
import json, numpy as np, os

def save_calib(mtx, dist, path="calibration/camera_calib.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {"mtx": mtx.tolist(), "dist": dist.tolist()}
    with open(path,"w") as f:
        json.dump(data, f, indent=2)

def load_calib(path="calibration/camera_calib.json"):
    import numpy as np
    with open(path,"r") as f:
        data = json.load(f)
    return np.array(data["mtx"]), np.array(data["dist"])

def save_homography(H, path="models/homography/homography.json"):
    import os, json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,"w") as f:
        json.dump({"H": H.tolist()}, f, indent=2)

def load_homography(path="models/homography/homography.json"):
    import json
    with open(path,"r") as f:
        data = json.load(f)
    import numpy as np
    return np.array(data["H"])
