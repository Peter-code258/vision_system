# Vision System — 多模态可见光 + 热红外视觉平台

> 一个端到端的视觉项目模版，包含：数据导入（COCO→YOLO）、训练（Ultralytics YOLO）、评估、ONNX/TensorRT 导出、PyQt5 本地面板、FastAPI + Vue 管理后台、热红外融合、摄像头标定与 Docker 部署。
---

## 目录（快速导航）

* [快速开始（Fast Start）](#快速开始Fast-Start)
* [环境准备（Ubuntu22 / Python / CUDA）](#环境准备Ubuntu22--Python--CUDA)
* [数据：COCO → YOLO 并自动 split](#数据COCO--YOLO-并自动-split)
* [训练（train）](#训练train)
* [评估（evaluate）](#评估evaluate)
* [ONNX / TensorRT 导出](#onnx--tensorrt-导出)
* [一键部署（deploy.sh）](#一键部署deploysh)
* [启动后端（FastAPI）与前端（Vue）](#启动后端FastAPI与前端Vue)
* [本地 GUI（PyQt5）](#本地-guipyqt5)
* [实时推理（REST / WebSocket / 单张图像）](#实时推理rest--websocket--单张图像)
* [摄像头标定与热红外融合](#摄像头标定与热红外融合)
* [Docker（CUDA12.8 + ONNXRuntime-GPU）](#dockercuda128--onnxruntime-gpu)
* [常见问题与排查（Troubleshooting）](#常见问题与排查Troubleshooting)
* [常用命令速查（Quick Commands）](#常用命令速查Quick-Commands)

---

## 快速开始 (Fast Start)

把仓库克隆到本机并进入项目根目录 `vision_system/`：

```bash
git clone <你的仓库地址>
cd vision_system
```

建议使用 Python 虚拟环境（示例使用 Python 3.10+）：

```bash
python3.10 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> 如果你要使用 GPU 推理（ONNXRuntime-GPU / PyTorch GPU），请参考下面的“环境准备”一节，按 CUDA 版本安装对应的 wheel。

---

## 环境准备（Ubuntu22 / Python / CUDA）

### 系统依赖（示例）

```bash
sudo apt update
sudo apt install -y build-essential python3-dev python3-venv git wget curl libgl1-mesa-glx libglib2.0-0
```

### Python 依赖（在已激活虚拟环境中）

```bash
pip install -r requirements.txt
```

### ONNXRuntime-GPU（可选）

如果你的机器安装了 CUDA（本项目参考 CUDA 12.8），请安装与 CUDA 匹配的 `onnxruntime-gpu`，例如：

```bash
pip install onnxruntime-gpu==1.18.0
```

### PyTorch（可选，安装时选对应 CUDA 版本）

示例（按官网选择合适索引）：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

> **注意**：不同 CUDA 版本对包版本要求不同，导出/推理失败时先确认 `torch` / `onnxruntime` 与系统 CUDA 驱动兼容。

---

## 数据：COCO → YOLO 并自动 split

脚本位置：

```
dataset/importers/coco_convert_and_split.py
```

**示例命令**（在项目根目录）：

```bash
python3 dataset/importers/coco_convert_and_split.py \
  --coco /path/to/instances_train2017.json \
  --images /path/to/train2017 \
  --out dataset/yolo \
  --split 0.8,0.1,0.1 \
  --seed 42
```

脚本结果：

```
dataset/yolo/
  images/{train,val,test}/
  labels/{train,val,test}/
```

并自动生成 `vision_system/configs/dataset.yaml`（Ultralytics 格式），可直接用于训练。

---

## 训练（train）

训练脚本（增强版）：

```
src/training/train.py
```

支持：

* 自动保存 `best.pt` 到 `models/exported/best.pt`（`--save_export`）
* 支持 resume（`--resume`）
* 可选 Weights & Biases 日志（`--wandb_project`）

**示例 - 基本训练**：

```bash
source venv/bin/activate
python3 src/training/train.py \
  --data configs/dataset.yaml \
  --pretrained yolov8m.pt \
  --epochs 80 \
  --imgsz 640 \
  --batch 16 \
  --save_export
```

**示例 - 恢复训练（resume）**：

```bash
python3 src/training/train.py --data configs/dataset.yaml --resume --save_export
```

**示例 - 使用 W&B**（需 `pip install wandb` 且设置 `WANDB_API_KEY`）：

```bash
export WANDB_API_KEY=your_key
python3 src/training/train.py --data configs/dataset.yaml --wandb_project "my-project" --save_export
```

训练产物位于：

```
runs/train/<exp>/
  weights/best.pt
  weights/last.pt
  results.csv
  ...
```

---

## 评估（evaluate）

评估脚本：

```
src/training/evaluate.py
```

快速运行（Python one-liner）：

```bash
python3 - <<'PY'
from ultralytics import YOLO
m = YOLO("models/exported/best.pt")
res = m.val()
print(res)
PY
```

---

## ONNX / TensorRT 导出

导出脚本：

```
tools/export_onnx.py
```

**导出 ONNX（FP16 可选）**：

```bash
python3 tools/export_onnx.py \
  --weights models/exported/best.pt \
  --output models/exported/best.onnx \
  --imgsz 640 \
  --half \
  --dynamic
```

如果要生成 TensorRT engine（需系统安装 TensorRT 并可用 `trtexec`）：

```
tools/trt_build.sh
```

示例：

```bash
bash tools/trt_build.sh models/exported/best.onnx models/exported/best_fp16.engine fp16
```

---

## 一键部署（export -> onnx -> upload -> start）

脚本：

```
tools/deploy.sh
```

**使用（默认后端服务需运行在 `http://localhost:8000`）**：

```bash
chmod +x tools/deploy.sh
./tools/deploy.sh
```

脚本逻辑：

1. 从 `models/exported/best.pt` 导出 `best.onnx`
2. 上传到后端 API `/upload_model/onnx`
3. 调用 `/set_backend` 切换为 `onnx`
4. 调用 `/start` 启动推理循环

---

## 启动后端（FastAPI）与前端（Vue）

### 后端（FastAPI）

入口：

```
src/server/main_api.py
```

**运行（开发模式）**：

```bash
uvicorn src.server.main_api:app --host 0.0.0.0 --port 8000 --reload
```

API 文档：

```
http://localhost:8000/docs
```

常用接口：

* `POST /upload_model/onnx` — 上传 ONNX 文件（multipart form）
* `POST /set_backend` — 设置 backend（`onnx` / `ultralytics`）
* `POST /start` — 启动推理循环（摄像头 + detector）
* `POST /stop` — 停止推理
* `GET /video_feed` — MJPEG 流
* `WS /ws` — WebSocket 推送每帧（base64 image + detections）

### 前端（Vue 3 + Vite）

前端目录：

```
web/
```

运行（开发）：

```bash
cd web
npm install
npm run dev
# 打开 http://localhost:5173
```

> Vite 已设置代理到 `http://localhost:8000`（参见 `vite.config.js`），Web 前端可以直接与后端交互。

---

## 本地 GUI（PyQt5）

主程序：

```
src/gui/pyqt_main.py
```

运行：

```bash
source venv/bin/activate
python3 src/gui/pyqt_main.py
```

功能：本地摄像头实时显示、启动/停止推理、切换后端、展示热图等。

---

## 实时推理（REST / WebSocket / 单张图像）

### 单张图像推理示例脚本（示例文件：`tools/infer_single_image.py`）

```python
# 示例（你也可以把它保存为 tools/infer_single_image.py）
from src.detectors.onnx_infer import ONNXDetector
import cv2
det = ONNXDetector("models/exported/best.onnx", input_size=640)
img = cv2.imread("samples/test.jpg")
dets, latency = det.infer(img)
print("latency:", latency, "dets:", dets)
# 可视化并保存
for d in dets:
  x1,y1,x2,y2 = d['box']; cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
cv2.imwrite("out.jpg", img)
```

### 通过后端启动实时推理

1. 启动后端 `uvicorn ...`
2. 上传 ONNX（或确保 `models/exported/best.onnx` 已存在并 STATE 指向）
3. 调用：

```bash
curl -X POST http://localhost:8000/start
```

4. 用前端或 WebSocket 订阅 `ws://localhost:8000/ws` 即可接收每帧图像与检测结果。

---

## 摄像头标定与热红外融合

标定脚本：

```
calibration/collect_chessboard.py
calibration/calibrate_camera.py
calibration/calibrate_and_align.py
```

* 先用 `collect_chessboard.py` 采集棋盘图像（按指示按 `s` 保存）。
* 使用 `calibrate_camera.py` 对采集结果标定并生成相机内参（保存到 `models/homography/`）。
* 使用 `calibrate_and_align.py` 生成 RGB ↔ thermal 的 Homography（`models/homography/homography.json`），供 `src/fusion/thermal_fusion.py` 使用。

热红外融合函数（已实现）：

```py
from src.fusion.thermal_fusion import fuse_rgb_and_thermal
fused_img, warped_thermal = fuse_rgb_and_thermal(rgb_bgr, thermal_img, H=H, alpha=0.45)
```

---

## Docker（CUDA 12.8 + ONNXRuntime-GPU）

提供 Docker 文件：

```
docker/Dockerfile.backend
docker/docker-compose.yml
```

构建并启动：

```bash
cd docker
docker-compose build
docker-compose up -d
```

容器会把项目目录挂载为 `/app`，FastAPI 暴露在 `8000` 端口。进入容器验证 GPU：

```bash
docker exec -it vision_backend bash
python3 - <<'PY'
import torch, onnxruntime as ort
print("torch.cuda:", torch.cuda.is_available())
print("onnxruntime device:", ort.get_device())
PY
```

---

## 常见问题与排查（Troubleshooting）

* **ONNX 导出失败**：先尝试不使用 `--half`，或降低 `opset`；确认 `ultralytics` 与 `torch` 版本匹配。
* **ONNX Runtime 无法使用 GPU**：安装 `onnxruntime-gpu`，并确认 CUDA、驱动和 cuDNN 匹配。
* **trtexec 报错**：检查 TensorRT 安装、驱动、`trtexec` 可执行权限；尝试调整动态 shape 或 workspace。
* **摄像头无法打开**：检查设备索引、权限（Docker 需映射 `/dev/video*`）或被其它程序占用。
* **显存不足**：减小 `--imgsz`、`--batch` 或使用更小模型（`yolov8n` / `yolov8s`），或启用 `--accumulate`（累积梯度）。

---

## 常用命令速查（Quick Commands）

```bash
# 激活环境
source venv/bin/activate

# COCO -> YOLO 并 split
python3 dataset/importers/coco_convert_and_split.py --coco /data/instances.json --images /data/images --out dataset/yolo

# 训练（带自动拷贝 best.pt）
python3 src/training/train.py --data configs/dataset.yaml --pretrained yolov8m.pt --epochs 80 --imgsz 640 --batch 16 --save_export

# 评估
python3 - <<'PY'
from ultralytics import YOLO
print(YOLO("models/exported/best.pt").val())
PY

# 导出 ONNX
python3 tools/export_onnx.py --weights models/exported/best.pt --output models/exported/best.onnx --imgsz 640 --half --dynamic

# TensorRT engine（需要 trtexec）
bash tools/trt_build.sh models/exported/best.onnx models/exported/best_fp16.engine fp16

# 一键 deploy (export->upload->set backend->start)
chmod +x tools/deploy.sh
./tools/deploy.sh

# 启动后端 (dev)
uvicorn src.server.main_api:app --reload --host 0.0.0.0 --port 8000

# 启动前端
cd web; npm install; npm run dev

# 启动本地 PyQt 面板
python3 src/gui/pyqt_main.py

# 启动 Docker 后端（示例）
cd docker
docker-compose up -d --build
```

---

## 贡献 & 许可证

* 欢迎提交 Issue / PR。
* 请在 `requirements.txt` 中锁定你实际使用的包版本以提高可重复性。
* 本仓库默认无许可证，请根据需要添加 `LICENSE` 文件（建议 MIT）。

---