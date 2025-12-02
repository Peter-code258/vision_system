# 👁️ Vision-System：多模态视觉融合系统

> **当可见光遇见热红外，当本地推理拥抱Web控制——这是一个集成了前沿技术与优雅设计的视觉系统，让你的视觉项目“看得更清、跑得更快、管得更顺”。**

## ✨ 系统亮点

- **🧠 多模态融合**：无缝结合可见光与热红外图像，获取超越单一模态的感知能力。
- **⚡ 极致加速**：支持从PyTorch到ONNX再到TensorRT的完整加速流水线，推理速度飞跃。
- **🖥️ 双端交互**：既提供直观的PyQt5桌面应用，也拥有现代化的FastAPI + Vue Web控制台。
- **🔧 开箱即用**：从数据准备、模型训练到部署应用，提供完整工具链与一键脚本。
- **📊 实验管理**：集成W&B、TensorBoard，训练过程透明可控，结果轻松追溯。

## 🚀 快速启动

### 1. 环境配置（Ubuntu 22.04）

```bash
# 1. 克隆项目仓库
git clone https://github.com/yourname/vision_system.git
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

## 📁 项目结构

```
vision_system/
├── backend/                 # FastAPI后端
├── frontend/vue-app/       # Vue前端
├── ui/                     # PyQt5桌面应用
├── training/               # 模型训练模块
├── inference/              # 推理模块
├── export/                 # 模型导出(ONNX/TensorRT)
├── fusion/                 # 多模态融合模块
├── calibration/            # 相机标定工具
├── datasets/               # 数据集处理
├── configs/                # 配置文件
├── models/weights/         # 模型权重存放处
├── deploy/                 # 部署脚本
└── utils/                  # 通用工具函数
```

## 🤝 贡献与使用

1. **自定义模型**：将你的 `.pt` 权重文件放入 `models/weights/`，即可在训练或推理中直接调用。
2. **扩展功能**：项目采用模块化设计，欢迎贡献新的融合算法、推理优化或界面改进。
3. **问题反馈**：遇到任何问题或有改进建议？欢迎提交Issue或Pull Request！

---

**星光不问赶路人，视觉不负有心人** —— 愿这个系统能加速你的视觉智能探索之旅！🚀

> 项目持续更新中，记得 `git pull` 获取最新功能！
