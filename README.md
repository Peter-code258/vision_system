
# 🚀 Vision System — 多模态可见光 + 热红外视觉平台

> **✨ 一个炫酷且功能完备的端到端视觉项目模版！从数据到部署，从本地界面到远程管理，让你轻松玩转多模态视觉AI。**

---

## 🗺️ 目录（快速导航）

* [⚡ 快速开始](#快速开始)
* [🛠️ 环境准备](#环境准备)
* [📊 数据准备：COCO → YOLO](#数据准备)
* [🏋️ 模型训练](#模型训练)
* [📈 模型评估](#模型评估)
* [⚙️ 模型导出：ONNX / TensorRT](#模型导出)
* [🚀 一键部署](#一键部署)
* [🌐 启动Web服务](#启动web服务)
* [🖥️ 本地图形界面](#本地图形界面)
* [🎯 实时推理](#实时推理)
* [🎨 摄像头标定与热红外融合](#摄像头标定与热红外融合)
* [🐳 Docker部署](#docker部署)
*  [❓ 常见问题](#常见问题)
* [📖 命令速查](#命令速查)

---

## 📁 项目结构

```
vision_system/                          <-- 项目根目录
├── README.md                           # 项目总体说明、快速启动、常见问题
├── requirements.txt                    # Python 依赖
├── setup_ubuntu22.sh                   # 一键在 Ubuntu22.04 上准备环境的脚本（引导）
├── package.json                        # 前端（Vue）依赖与脚本
├── .env.sample                         # 环境变量示例（端口、摄像头索引等）
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
│   ├── api_clients/                    # JS/Python 客户端封装（调用后端）
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
├── docker/                             # Dockerfile / docker-compose 示例
│   ├── Dockerfile.backend
│   └── docker-compose.yml
│
└── logs/
    ├── server.log
    └── runs/                           # 训练 / 导出产生的多份 runs 目录

```

## 快速开始

1.  **克隆项目**：把仓库“搬”到你的电脑上。
    ```bash
    git clone git clone https://github.com/Peter-code258/vision_system.git
    cd vision_system
    ```

2.  **准备环境**：建议创建一个独立的Python虚拟环境。
    ```bash
    python3.10 -m venv venv
    source venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

> **小贴士**：如果你想使用GPU进行加速推理，请继续阅读下面的“环境准备”部分，安装对应版本的PyTorch和ONNXRuntime。

---

 ## 环境准备

### 安装系统依赖
```bash
sudo apt update
sudo apt install -y build-essential python3-dev python3-venv git wget curl libgl1-mesa-glx libglib2.0-0
```

### 安装Python依赖
确保已在虚拟环境中，然后执行：
```bash
pip install -r requirements.txt
```

### 安装GPU支持（可选但推荐）
- **安装ONNXRuntime-GPU**（以CUDA 12.8为例）：
    ```bash
    pip install onnxruntime-gpu==1.18.0
    ```
- **安装PyTorch（GPU版）**：
    ```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    ```

> **⚠️ 重要提示**：请确保你安装的`torch`、`onnxruntime`版本与你系统上的CUDA驱动版本兼容，这是许多导出或推理错误的根源！

---

## 📊 数据准备

我们提供了将标准COCO数据集转换为YOLO格式并自动分割的脚本。

**脚本位置**：`dataset/importers/coco_convert_and_split.py`

**使用示例**：
```bash
python3 dataset/importers/coco_convert_and_split.py \
  --coco /path/to/instances_train2017.json \
  --images /path/to/train2017 \
  --out dataset/yolo \
  --split 0.8,0.1,0.1 \
  --seed 42
```

转换后，你会得到结构清晰的YOLO格式数据集，并且配置文件`vision_system/configs/dataset.yaml`会自动生成，可以直接用于训练！

---

## 🏋️ 模型训练

**训练脚本**：`src/training/train.py`
这个增强版脚本支持自动保存最佳模型、断点续训和W&B日志。

- **基础训练命令**：
    ```bash
    python3 src/training/train.py \
      --data configs/dataset.yaml \
      --pretrained yolov8m.pt \
      --epochs 80 \
      --imgsz 640 \
      --batch 16 \
      --save_export
    ```
- **恢复训练**：
    ```bash
    python3 src/training/train.py --data configs/dataset.yaml --resume --save_export
    ```
- **使用W&B记录实验**（需先设置API Key）：
    ```bash
    export WANDB_API_KEY=your_key
    python3 src/training/train.py --data configs/dataset.yaml --wandb_project “my-project” --save_export
    ```

所有训练成果（模型、日志等）都会保存在`runs/train/<exp>/`目录下。

---

## 📈 模型评估

使用内置脚本或一行Python命令快速评估模型性能。

**快速评估**：
```bash
python3 - <<‘PY’
from ultralytics import YOLO
m = YOLO(“models/exported/best.pt”)
res = m.val()
print(res)
PY
```

---

## ⚙️ 模型导出

将训练好的PyTorch模型导出为高性能的ONNX或TensorRT格式。

- **导出ONNX模型**：
    ```bash
    python3 tools/export_onnx.py \
      --weights models/exported/best.pt \
      --output models/exported/best.onnx \
      --imgsz 640 \
      --half \
      --dynamic
    ```
- **构建TensorRT引擎**（需系统已安装TensorRT）：
    ```bash
    bash tools/trt_build.sh models/exported/best.onnx models/exported/best_fp16.engine fp16
    ```

---

## 🚀 一键部署

我们提供了一个超方便的部署脚本`tools/deploy.sh`，它能自动完成模型导出、上传到后端、切换推理引擎并启动服务的全过程！

**使用方法**：
```bash
chmod +x tools/deploy.sh
./tools/deploy.sh
```

---

## 🌐 启动Web服务

### 后端（FastAPI）
启动高性能的API后端服务：
```bash
uvicorn src.server.main_api:app --host 0.0.0.0 --port 8000 --reload
```
启动后，可以访问 `http://localhost:8000/docs` 查看完整的API交互文档。

### 前端（Vue 3 + Vite）
启动现代化的管理前端：
```bash
cd web
npm install
npm run dev
```
打开浏览器访问 `http://localhost:5173` 即可。

---

## 🖥️ 本地图形界面

我们还准备了功能丰富的本地PyQt5图形界面！

**启动方式**：
```bash
source venv/bin/activate
python3 src/gui/pyqt_main.py
```
在这里，你可以进行本地摄像头实时预览、启停推理、切换后端、查看热图等操作。

---

## 🎯 实时推理

系统支持多种灵活的推理方式：
- **通过后端API启动**：上传模型后，调用`/start`接口即可。
- **单张图片测试**：我们提供了示例脚本(`tools/infer_single_image.py`)。
- **实时视频流**：可以通过WebSocket订阅`ws://localhost:8000/ws`，实时获取每一帧的检测结果。

---

## 🎨 摄像头标定与热红外融合

要实现精准的多模态融合，首先要进行摄像头标定和对齐。

1.  **采集标定板图像**：运行 `calibration/collect_chessboard.py`，按提示操作。
2.  **计算相机参数**：使用 `calibration/calibrate_camera.py` 进行标定。
3.  **计算对齐矩阵**：运行 `calibration/calibrate_and_align.py` 获取RGB与热红外图像的对齐关系。

完成以上步骤后，就可以在代码中轻松调用融合函数了：
```python
from src.fusion.thermal_fusion import fuse_rgb_and_thermal
fused_img, warped_thermal = fuse_rgb_and_thermal(rgb_bgr, thermal_img, H=H, alpha=0.45)
```

---

## 🐳 Docker部署

我们提供了配置好CUDA环境的Dockerfile和docker-compose文件，让你可以快速构建和启动一个包含GPU支持的标准化服务环境。

**构建并启动**：
```bash
cd docker
docker-compose build
docker-compose up -d
```

**验证GPU在容器内是否可用**：
```bash
docker exec -it vision_backend bash
python3 - <<‘PY’
import torch, onnxruntime as ort
print(“torch.cuda:”, torch.cuda.is_available())
print(“onnxruntime device:”, ort.get_device())
PY
```

---

## ❓ 常见问题

| 问题现象 | 可能原因与排查建议 |
| :--- | :--- |
| **ONNX导出失败** | 尝试不加`--half`参数，或降低`opset`版本；检查`ultralytics`和`torch`版本是否匹配。 |
| **ONNX Runtime无法使用GPU** | 确认安装的是`onnxruntime-gpu`；检查CUDA、驱动和cuDNN版本是否匹配。 |
| **TensorRT转换(trtexec)报错** | 检查TensorRT安装和驱动；尝试调整动态shape或workspace大小。 |
| **摄像头打不开** | 检查设备索引号、用户权限（Docker需要映射`/dev/video*`设备），或是否被其他程序占用。 |
| **显存不足(OOM)** | 尝试减小输入图像尺寸(`--imgsz`)、批次大小(`--batch`)，或使用更小的模型(如`yolov8n`)。 |

---

## 📖 命令速查

| 功能 | 命令 |
| :--- | :--- |
| **激活虚拟环境** | `source venv/bin/activate` |
| **转换COCO数据集** | `python3 dataset/importers/coco_convert_and_split.py …` |
| **训练模型** | `python3 src/training/train.py --data configs/dataset.yaml …` |
| **评估模型** | `python3 -c “from ultralytics import YOLO; print(YOLO(‘models/exported/best.pt’).val())”` |
| **导出ONNX** | `python3 tools/export_onnx.py --weights models/exported/best.pt …` |
| **一键部署** | `chmod +x tools/deploy.sh && ./tools/deploy.sh` |
| **启动后端** | `uvicorn src.server.main_api:app --reload --host 0.0.0.0 --port 8000` |
| **启动前端** | `cd web; npm install; npm run dev` |
| **启动PyQt5界面** | `python3 src/gui/pyqt_main.py` |
| **启动Docker服务** | `cd docker; docker-compose up -d --build` |

---

**🌟 欢迎贡献！** 如果你有好的想法或发现了问题，欢迎提交Issue或Pull Request。让我们共同打造更强大的视觉系统！
