# 👁️ Vision-System：多模态视觉融合系统

> **当可见光遇见热红外，当本地推理拥抱Web控制——这是一个集成了前沿技术与优雅设计的视觉系统，让你的视觉项目“看得更清、跑得更快、管得更顺”。**

## ✨ 系统亮点

- **🧠 多模态融合**：无缝结合可见光与热红外图像，获取超越单一模态的感知能力。
- **⚡ 极致加速**：支持从PyTorch到ONNX再到TensorRT的完整加速流水线，推理速度飞跃。
- **🖥️ 双端交互**：既提供直观的PyQt5桌面应用，也拥有现代化的FastAPI + Vue Web控制台。
- **🔧 开箱即用**：从数据准备、模型训练到部署应用，提供完整工具链与一键脚本。
- **📊 实验管理**：集成W&B、TensorBoard，训练过程透明可控，结果轻松追溯。

## 📁 项目结构
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

## 🚀 快速启动

### 1. 环境配置（Ubuntu 22.04）

```bash
# 1. 克隆项目仓库
git clone https://github.com/Peter-code258/vision_system.git
cd vision_system

# 2. 创建并激活Python虚拟环境
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip -y
python3.10 -m venv venv
source venv/bin/activate

# 3. 安装依赖包
pip install --upgrade pip
pip install -r requirements.txt

# 4. 安装PyTorch（请根据你的CUDA版本选择）
# 以CUDA 11.8为例：
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. 数据准备：COCO数据集转换

1. 将你的COCO数据集按以下结构放置：
   ```
   datasets/coco/
   ├── train2017/      # 训练图像
   ├── val2017/        # 验证图像
   └── annotations/    # 标注文件
   ```

2. 一键转换为YOLO格式：
   ```bash
   python datasets/coco_to_yolo.py \
       --coco_dir datasets/coco \
       --output_dir datasets/yolo_coco \
       --train_ratio 0.8
   ```
   转换完成后，`datasets/yolo_coco/dataset.yaml` 将直接用于训练！

## 🎯 核心功能

### 🏋️ 模型训练

```bash
# 基础训练
python training/train.py \
    --data datasets/yolo_coco/dataset.yaml \
    --model configs/model/yolov8.yaml \
    --epochs 100 \
    --batch 16 \
    --img 640

# 恢复训练（上次意外中断？没问题！）
python training/train.py --resume runs/train/exp/last.pt

# 使用W&B记录实验（需设置API密钥）
WANDB_API_KEY=your_key_here python training/train.py --wandb
```

**训练成果**：每个实验都会在 `runs/train/` 下生成完整记录，包括最佳模型、最后检查点、训练指标和TensorBoard日志。

### 🔮 推理演示

无论你是想测试单张图片、连接本地摄像头还是接入网络视频流，我们都准备好了：

```bash
# 单张图像推理
python inference/vision_inference.py \
    --weights runs/train/exp/best.pt \
    --source samples/test.jpg

# 本地摄像头实时推理
python inference/vision_inference.py --weights runs/train/exp/best.pt --source 0

# RTSP视频流推理
python inference/vision_inference.py \
    --weights runs/train/exp/best.pt \
    --source rtsp://your_stream_url
```

### ⚡ 模型加速与导出

追求极致速度？我们的加速流水线能帮你把模型性能榨干：

```bash
# 1. 导出为ONNX格式
python export/export_onnx.py \
    --weights runs/train/exp/best.pt \
    --output exports/best.onnx

# 2. 转换为TensorRT引擎（FP16精度提速）
python export/export_trt.py \
    --onnx exports/best.onnx \
    --output exports/best_fp16.trt \
    --fp16
```

**一键部署神器**：`bash deploy/deploy.sh runs/train/exp/best.pt`  
这个脚本会自动完成模型导出、转换、上传和重启服务的全过程！

## 🎨 交互界面

### 桌面应用 (PyQt5)

启动炫酷的本地图形界面：
```bash
python ui/pyqt_app.py
```
**功能一览**：实时摄像头显示、热红外融合可视化、推理参数调节、模型热切换、标定工具快捷入口。

### Web控制台 (FastAPI + Vue)

**后端启动**：
```bash
uvicorn backend.fastapi_app:app --host 0.0.0.0 --port 8000
```

**前端启动**：
```bash
cd frontend/vue-app
npm install
npm run dev
```
访问 `http://localhost:5173` 即可享受现代化的远程控制体验！

## 🔧 高级工具

### 摄像头标定
```bash
# 采集标定图像
python calibration/calibrate_camera.py --capture

# 计算标定参数
python calibration/calibrate_camera.py --calibrate \
    --images calibration/captured/
```
获得精确的相机内参和畸变参数，让每个像素都物尽其用。

### 热红外融合
```bash
python fusion/thermal_fusion.py \
    --rgb samples/rgb.jpg \
    --thermal samples/thermal.png \
    --mode additive
```
探索可见光与热红外的融合魔法，系统支持多种融合模式，满足不同场景需求。

## 🤝 贡献与使用

1. **自定义模型**：将你的 `.pt` 权重文件放入 `models/weights/`，即可在训练或推理中直接调用。
2. **扩展功能**：项目采用模块化设计，欢迎贡献新的融合算法、推理优化或界面改进。
3. **问题反馈**：遇到任何问题或有改进建议？欢迎提交Issue或Pull Request！

---

**星光不问赶路人，视觉不负有心人** —— 愿这个系统能加速你的视觉智能探索之旅！🚀

> 项目持续更新中，记得 `git pull` 获取最新功能！
