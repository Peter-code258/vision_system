---

# Vision-System: 多模态可见光 + 热红外 + PyQt5 + FastAPI/Vue + TensorRT 加速视觉系统

本项目提供一个完整可运行的多模态视觉系统，包含：

* COCO 数据集自动导入与转换
* 可见光 / 热红外 双模态融合
* PyTorch 模型训练（resume/W&B/best 自动保存）
* ONNX/TensorRT 加速推理
* PyQt5 桌面应用
* FastAPI + Vue 的远程 Web 控制台
* 摄像头标定工具
* 自动化部署流水线 `deploy.sh`

本 README 提供从 0 到可运行的全部命令。

---

## 1. 环境安装 (Ubuntu 22)

### 1.1 克隆项目

```bash
git clone https://github.com/yourname/vision_system.git
cd vision_system
```

### 1.2 安装 Python 环境

```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip -y

python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 1.3 安装 PyTorch (根据你 GPU 的 CUDA 版本)

以 CUDA 11.8 为例：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 2. COCO 数据集导入与转换

项目内置 COCO → YOLO 转换脚本：

```
datasets/
│── coco/
│── coco_to_yolo.py
```

### 2.1 将原始 COCO 放入 datasets/coco

例如：

```
datasets/coco/train2017/
datasets/coco/val2017/
datasets/coco/annotations/
```

### 2.2 执行转换 + 自动 split

```bash
python datasets/coco_to_yolo.py \
    --coco_dir datasets/coco \
    --output_dir datasets/yolo_coco \
    --train_ratio 0.8
```

转换后生成结构：

```
datasets/yolo_coco/
│── images/train/
│── images/val/
│── labels/train/
│── labels/val/
│── dataset.yaml
```

你可以用 dataset.yaml 直接训练。

---

## 3. 训练

训练脚本位置：

```
training/train.py
```

核心功能包括：

* 自动保存 best.pt
* resume 训练
* W&B 日志
* 本地日志（TensorBoard 格式）
* 自动记录配置

### 3.1 基本训练指令

```bash
python training/train.py \
    --data datasets/yolo_coco/dataset.yaml \
    --model configs/model/yolov8.yaml \
    --epochs 100 \
    --batch 16 \
    --img 640
```

### 3.2 恢复训练 resume

```bash
python training/train.py --resume runs/train/exp/last.pt
```

### 3.3 启用 W&B

```bash
WANDB_API_KEY=xxxx python training/train.py --wandb
```

### 3.4 指定输出目录

```bash
python training/train.py --project runs/train/custom_exp
```

训练结果会输出：

```
runs/train/exp/
│── best.pt
│── last.pt
│── results.csv
│── tensorboard/
```

---

## 4. 推理（单张图像 or RTSP 摄像头）

推理脚本位置：

```
inference/vision_inference.py
```

### 4.1 单张图像推理

```bash
python inference/vision_inference.py \
    --weights runs/train/exp/best.pt \
    --source samples/test.jpg \
    --img 640
```

### 4.2 摄像头推理

```bash
python inference/vision_inference.py \
    --weights runs/train/exp/best.pt \
    --source 0
```

### 4.3 RTSP 推理

```bash
python inference/vision_inference.py \
    --weights runs/train/exp/best.pt \
    --source rtsp://xxx
```

---

## 5. ONNX / TensorRT 导出

导出脚本：

```
export/export_onnx.py
export/export_trt.py
```

### 5.1 导出 ONNX

```bash
python export/export_onnx.py \
    --weights runs/train/exp/best.pt \
    --output exports/best.onnx \
    --img 640
```

### 5.2 导出 TensorRT（FP16）

```bash
python export/export_trt.py \
    --onnx exports/best.onnx \
    --output exports/best_fp16.trt \
    --fp16
```

---

## 6. 一键部署流水线 deploy.sh

脚本路径：

```
deploy/deploy.sh
```

功能：

1. 导出 ONNX
2. 转 TensorRT
3. 上传到 FastAPI 后端
4. 通知后端重启推理

使用方法：

```bash
bash deploy/deploy.sh runs/train/exp/best.pt
```

---

## 7. PyQt5 图形界面

主程序位置：

```
ui/pyqt_app.py
```

启动方法：

```bash
python ui/pyqt_app.py
```

界面功能：

* 摄像头实时显示
* 热红外融合
* 推理参数调节
* 模型切换
* 标定工具入口

---

## 8. FastAPI + Vue Web 控制台

后台目录：

```
backend/
    fastapi_app.py
frontend/
    vue-app/
```

### 8.1 启动 FastAPI

```bash
uvicorn backend.fastapi_app:app --host 0.0.0.0 --port 8000
```

### 8.2 启动 Vue

```bash
cd frontend/vue-app
npm install
npm run dev
```

浏览器访问：

```
http://localhost:5173
```

---

## 9. 摄像头标定

标定工具路径：

```
calibration/calibrate_camera.py
```

### 9.1 标定采集

```bash
python calibration/calibrate_camera.py --capture
```

### 9.2 执行标定

```bash
python calibration/calibrate_camera.py --calibrate \
    --images calibration/captured/
```

会输出：

```
calibration/output/camera_intrinsics.yaml
calibration/output/camera_distortion.yaml
```

---

## 10. 热红外融合

融合模块：

```
fusion/thermal_fusion.py
```

独立运行测试：

```bash
python fusion/thermal_fusion.py \
    --rgb samples/rgb.jpg \
    --thermal samples/thermal.png \
    --mode additive
```

项目中 PyQt5、FastAPI 均自动调用该模块。

---

## 11. 模型导入（自定义权重）

将你的模型放入：

```
models/weights/
```

例：

```
models/weights/custom_model.pt
```

推理时：

```bash
python inference/vision_inference.py --weights models/weights/custom_model.pt
```

训练时：

```bash
python training/train.py --weights models/weights/custom_model.pt
```

---

## 12. 项目目录结构（总览）

```
vision_system/                          <-- 项目根目录
├── README.md                           # 项目总体说明、快速启动、常见问题
├── requirements.txt                    # Python 依赖
├── setup_ubuntu22.sh                   # 一键在 Ubuntu22.04 上准备环境的脚本（引导）
├── package.json                        # 前端（Vue）依赖与脚本
├── .env.sample                         # 可选：环境变量示例（端口、摄像头索引等）
├── configs/
│   ├── default.yaml                    # 全局默认配置（device, confidence, input_size 等）
│   ├── dataset.yaml                    # 训练用 dataset 配置 (Ultralytics 格式)
│   ├── train.yaml                      # 训练超参（epochs, batch, lr 等）
│   └── inference.yaml                  # 推理参数（onnx/trt 路径、heatmap 等）
│
├── models/
│   ├── exported/
│   │   ├── best.pt                     # 训练得到的权重（示例位置）
│   │   ├── best.onnx                   # ONNX 导出文件（示例位置）
│   │   └── best_fp16.engine            # TensorRT engine（示例位置）
│   └── homography/
│       └── homography.json             # RGB <- thermal homography（配准文件）
│
├── dataset/
│   ├── yolo/                           # 目标 YOLO 格式数据（images/labels）
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── val/
│   │   └── labels/
│   │       ├── train/
│   │       └── val/
│   └── importers/
│       ├── coco2yolo.py                # COCO -> YOLO 转换脚本
│       ├── voc2yolo.py                 # VOC -> YOLO 转换脚本
│       ├── split_yolo.py               # 划分 train/val 脚本
│       └── verify_labels.py            # 检查标签可视化脚本
│
├── tools/
│   ├── export_onnx.py                  # 基于 ultralytics 的 ONNX 导出脚本
│   ├── trt_build.sh                    # trtexec/TensorRT 转换脚本示例
│   └── deploy.sh                       # 一键 deploy: export->onnx->upload->start（示例）
│
├── calibration/
│   ├── collect_chessboard.py           # 采集棋盘图像工具（交互）
│   ├── calibrate_camera.py             # OpenCV 相机标定脚本（保存 mtx/dist）
│   └── calibrate_and_align.py          # RGB<->Thermal 配对采集与手动配准生成 homography
│
├── src/
│   ├── api_clients/                    # 可选：JS/Python 客户端封装（调用后端）
│   │   └── backend_client.py
│   │
│   ├── detectors/                      # 各类后端推理器（统一接口）
│   │   ├── onnx_infer.py               # ONNX Runtime 推理器（完整预/后处理 + NMS）
│   │   ├── trt_infer.py                # TensorRT 推理器（engine loader + infer skeleton）
│   │   └── ultralytics_infer.py        # 直接调用 ultralytics 的推理器（训练/导出阶段备用）
│   │
│   ├── fusion/
│   │   └── thermal_fusion.py           # 热像→RGB 对齐、伪彩 & 叠加、ROI 温度统计
│   │
│   ├── sensors/
│   │   ├── thermal_reader.py           # 读取热像相机或热像视频（灰度归一化）
│   │   └── ir_reader.py                # 串口 IR（PIR/DIST/TEMP）读取器（线程、安全）
│   │
│   ├── utils/
│   │   ├── draw.py                     # 绘制检测框、热度条、ROI 信息
│   │   ├── camera_calib_io.py          # 保存/加载相机内参（mtx/dist）与 homography
│   │   ├── config_loader.py            # YAML 配置加载器（全局统一）
│   │   └── logger.py                   # 简易日志工具（写文件/控制台）
│   │
│   ├── server/
│   │   ├── main_api.py                 # FastAPI 管理后台（静态页面、模型上传、start/stop、train/eval）
│   │   ├── ws_stream.py                # WebSocket 帧+检测推送实现（JSON + base64 image）
│   │   └── mjpeg_stream.py             # MJPEG 生成器（/video_feed）
│   │
│   ├── training/
│   │   ├── train.py                    # 训练脚本（Ultralytics API 封装，支持 resume/wandb）
│   │   └── evaluate.py                 # 评估脚本（model.val() 结果封装 JSON）
│   │
│   ├── gui/
│   │   ├── pyqt_main.py                # PyQt5 控制面板主入口（嵌入视频、开关、参数面板）
│   │   └── qt_video_widget.py          # QLabel/QImage 显示抽象（高帧率显示帮助）
│   │
│   └── inference/
│       └── runner.py                   # 抽象的推理运行器：读取摄像头->推理->融合->结果回调（供 GUI/Server 调用）
│
├── web/                                # 前端 (Vue 3 + Vite)
│   ├── package.json
│   ├── index.html
│   ├── vite.config.js
│   └── src/
│       ├── main.js
│       ├── App.vue
│       ├── styles.css
│       └── components/
│           ├── TopBar.vue
│           ├── ModelManager.vue
│           ├── InferenceControls.vue
│           ├── HeatmapSettings.vue
│           ├── StreamCanvas.vue
│           └── LogsPanel.vue
│
├── docker/                             # 可选：Dockerfile / docker-compose 示例
│   ├── Dockerfile.backend
│   └── docker-compose.yml
│
└── logs/
    ├── server.log
    └── runs/                           # 训练 / 导出产生的多份 runs 目录

```
