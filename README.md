# YOLOv8n COCO2017 目标检测复现项目

本项目基于 Ultralytics YOLOv8n，在 COCO2017 数据集上完成目标检测模型训练与验证，记录训练参数、逐轮指标、模型权重、评估曲线及预测可视化结果。

## 1. 项目目录结构

```text
YOLOv8/
├── README.md
├── .gitignore
├── configs/
│   └── coco.yaml
├── datasets/
│   └── coco/
│       ├── annotations/
│       ├── images/
│       │   ├── train2017/
│       │   └── val2017/
│       └── labels/
│           ├── train2017/
│           └── val2017/
├── weights/
│   └── yolov8n.pt
├── train/
│   ├── args.yaml
│   ├── results.csv
│   ├── results.png
│   ├── YOLOv8n_100轮训练指标摘要.txt
│   ├── YOLOv8n_学习率变化曲线_100轮.png
│   ├── YOLOv8n_网络结构示意图.png
│   ├── YOLOv8n_训练集损失曲线_100轮.png
│   ├── train_batch*.jpg
│   └── weights/
│       ├── best.pt
│       └── last.pt
├── val/
│   ├── BoxF1_curve.png
│   ├── BoxPR_curve.png
│   ├── BoxP_curve.png
│   ├── BoxR_curve.png
│   ├── confusion_matrix.png
│   ├── confusion_matrix_normalized.png
│   ├── YOLOv8n_验证集mAP曲线_100轮.png
│   ├── YOLOv8n_验证集损失曲线_100轮.png
│   ├── YOLOv8n_验证集精确率与召回率曲线_100轮.png
│   └── val_batch*.jpg
├── test/
├── environment.yml
└── scripts/
    ├── train_coco2017.sh
    ├── validate_coco2017.sh
    └── test_coco2017.sh
```

目录说明：

- `configs/`：COCO2017 数据集配置文件。
- `datasets/`：COCO2017 图片、原始标注和 YOLO 格式标签。
- `weights/`：训练前使用的 YOLOv8n 预训练权重。
- `train/`：训练参数、逐轮训练记录、训练可视化和模型权重。
- `val/`：验证集评估曲线、混淆矩阵及预测可视化。
- `test/`：独立测试结果预留目录，目前没有独立测试集。
- `scripts/`：训练、验证和测试脚本。
- `environment.yml`：Conda 环境配置。

## 2. 实验环境

| 项目 | 版本或配置 |
|---|---|
| 操作系统 | Ubuntu 20.04.6 LTS |
| Python | 3.11.15 |
| PyTorch | 2.4.0 |
| Torchvision | 0.19.0 |
| CUDA Runtime | 12.1 |
| Ultralytics | 8.4.53 |
| NumPy | 1.26.4 |
| OpenCV | 4.10.0.84 |
| Matplotlib | 3.10.9 |
| GPU | NVIDIA A10 |
| 本次训练使用 GPU | 物理 GPU 2 |

本次训练通过 `CUDA_VISIBLE_DEVICES=2` 指定物理 GPU 2，因此程序内部的 `device=0` 对应所选物理 GPU。

### 2.1 创建环境

```bash
conda env create -f environment.yml
conda activate yolo311
```

如果环境已经存在，可以直接激活：

```bash
conda activate yolo311
```

### 2.2 Ultralytics 源码说明

本次实验使用的 Ultralytics 源码目录：

```text
/home/kang/fire1/ultralytics-main/
```

当前环境采用 editable install，原始依赖记录为：

```text
-e /home/kang/fire1/ultralytics-main
```

为了完整复现实验，需要保留对应的 Ultralytics 8.4.53 源码。迁移到其他机器时，应将源码目录一并保存，并根据新机器的实际路径重新安装：

```bash
pip install -e /path/to/ultralytics-main
```

上述服务器绝对路径不能直接作为其他机器上的通用路径。

## 3. 数据集说明

本项目使用 COCO2017 目标检测数据集，包含 80 个目标类别。

### 3.1 数据划分

| 数据划分 | 图片数量 | YOLO 标签文件数量 | 用途 |
|---|---:|---:|---|
| train2017 | 118,287 | 117,266 | 模型训练 |
| val2017 | 5,000 | 4,952 | 模型验证 |

部分图片不包含目标实例，因此没有对应的 YOLO 标签文件。这属于正常情况，不代表数据集缺失。

数据目录：

```text
datasets/coco/
├── annotations/
├── images/
│   ├── train2017/
│   └── val2017/
└── labels/
    ├── train2017/
    └── val2017/
```

### 3.2 数据集配置

配置文件为 `configs/coco.yaml`，主要配置如下：

```yaml
path: /home/kang/experiments/neural_reproduction/YOLOv8/datasets/coco
train: images/train2017
val: images/val2017
nc: 80
```

类别名称按照 COCO 官方 80 类顺序配置。

迁移项目时，需要将 `path` 修改为新机器上 COCO 数据集的实际路径。

### 3.3 训练集、验证集和测试集

- 训练集 `train2017` 用于模型参数学习。
- 验证集 `val2017` 用于训练期间的模型评估和指标计算。
- 独立测试集应当与训练集、验证集相互独立，用于最终泛化性能评估。

本项目实际使用了 COCO2017 的 `train2017` 和 `val2017`，没有配置独立的带标注测试集。因此，本次实验中的 Precision、Recall、mAP 等指标均为验证集指标，不能将其表述为独立测试集指标。

`test/` 和测试脚本仅作为后续扩展预留。

## 4. 模型与训练配置

### 4.1 模型信息

| 项目 | 配置 |
|---|---|
| 模型 | YOLOv8n |
| 任务 | 目标检测 |
| 预训练权重 | `weights/yolov8n.pt` |
| 数据集 | COCO2017 |
| 检测类别数 | 80 |

### 4.2 主要训练参数

| 参数 | 配置 |
|---|---|
| epochs | 100 |
| imgsz | 640 |
| batch | 16 |
| device | 0（映射到物理 GPU 2） |
| workers | 8 |
| amp | False |
| pretrained | True |
| plots | True |
| save | True |

训练实际使用的完整参数保存在：

```text
train/args.yaml
```

训练逐轮记录保存在：

```text
train/results.csv
```

训练曲线保存在：

```text
train/results.png
```

`results.csv` 包含 100 轮训练记录，既包含训练损失，也包含训练期间的验证指标；`results.png` 是相应训练和验证指标的可视化曲线。这两项均不是独立测试集结果。

## 5. 实验结果

本次 YOLOv8n 模型完成了 100 轮训练。

### 5.1 第 100 轮验证指标

| 指标 | 数值 |
|---|---:|
| Precision | 0.62740 |
| Recall | 0.47196 |
| mAP50 | 0.51276 |
| mAP50-95 | 0.36211 |

换算为百分比：

| 指标 | 百分比 |
|---|---:|
| Precision | 62.740% |
| Recall | 47.196% |
| mAP50 | 51.276% |
| mAP50-95 | 36.211% |

### 5.2 训练期间最佳指标

| 指标 | 最佳数值 | 对应轮次 |
|---|---:|---:|
| Precision | 0.63203 | 90 |
| Recall | 0.47306 | 99 |
| mAP50 | 0.51276 | 100 |
| mAP50-95 | 0.36211 | 100 |

以上均为训练期间在 COCO2017 验证集上得到的指标。

### 5.3 模型权重

训练结束后保存了两个模型权重：

```text
train/weights/best.pt
train/weights/last.pt
```

- `best.pt`：训练过程中保存的最佳模型权重。
- `last.pt`：最后一轮训练结束时保存的模型权重。

推荐使用 `best.pt` 进行后续验证和推理。

## 6. 结果文件说明

### 6.1 train/

| 文件 | 说明 |
|---|---|
| `args.yaml` | 训练参数记录 |
| `results.csv` | 100 轮训练及验证指标记录 |
| `results.png` | 训练与验证指标综合曲线 |
| `YOLOv8n_100轮训练指标摘要.txt` | 实验指标摘要 |
| `YOLOv8n_学习率变化曲线_100轮.png` | 学习率变化曲线 |
| `YOLOv8n_网络结构示意图.png` | 模型网络结构示意图 |
| `YOLOv8n_训练集损失曲线_100轮.png` | 训练集损失曲线 |
| `train_batch*.jpg` | 训练样本及标注可视化 |
| `weights/best.pt` | 最佳模型 |
| `weights/last.pt` | 最后一轮模型 |

### 6.2 val/

| 文件 | 说明 |
|---|---|
| `BoxP_curve.png` | Box Precision 曲线 |
| `BoxR_curve.png` | Box Recall 曲线 |
| `BoxF1_curve.png` | Box F1 曲线 |
| `BoxPR_curve.png` | Box PR 曲线 |
| `confusion_matrix.png` | 混淆矩阵 |
| `confusion_matrix_normalized.png` | 归一化混淆矩阵 |
| `YOLOv8n_验证集mAP曲线_100轮.png` | 验证集 mAP 曲线 |
| `YOLOv8n_验证集损失曲线_100轮.png` | 验证集损失曲线 |
| `YOLOv8n_验证集精确率与召回率曲线_100轮.png` | 验证集 Precision 与 Recall 曲线 |
| `val_batch*_labels.jpg` | 验证集真实标注可视化 |
| `val_batch*_pred.jpg` | 验证集预测结果可视化 |

上述验证图表主要由训练期间的验证过程生成，并非单独测试集评估结果。

## 7. 运行方法

运行脚本前，请确认：

1. 已激活 `yolo311` 环境。
2. COCO2017 图片和标签已放置在 `datasets/coco/`。
3. `configs/coco.yaml` 中的数据集路径正确。
4. `weights/yolov8n.pt` 预训练权重存在。
5. GPU 编号已根据当前机器情况调整。

### 7.1 模型训练

```bash
bash scripts/train_coco2017.sh
```

训练脚本默认：

- 使用物理 GPU 2。
- 训练 100 轮。
- 输入尺寸 640。
- Batch size 为 16。
- 训练结果输出至项目根目录下的 `train/`。

当前 `train/` 已保存本次实验结果。脚本会检测已有结果并停止，防止意外覆盖。重新训练前应先备份或整理已有结果。

### 7.2 验证

```bash
bash scripts/validate_coco2017.sh
```

默认使用：

```text
train/weights/best.pt
```

验证数据来自：

```text
configs/coco.yaml
```

验证结果输出到：

```text
val/
```

当前 `val/` 已有本次训练产生的验证结果。脚本会检测已有文件并停止，防止覆盖。

如需指定其他模型权重，可以设置：

```bash
MODEL_PATH=/path/to/best.pt bash scripts/validate_coco2017.sh
```

### 7.3 独立测试

只有在准备好独立测试集及其 YAML 配置后，才能运行：

```bash
TEST_DATA_YAML=/path/to/test.yaml bash scripts/test_coco2017.sh
```

模型默认使用：

```text
train/weights/best.pt
```

测试结果输出到：

```text
test/
```

注意：当前项目没有独立测试集。未提供 `TEST_DATA_YAML` 时，脚本会主动退出，不会将 `val2017` 当作测试集。

## 8. 依赖清单

以下为本次实验原始 `requirements.txt` 的完整内容，保留原始依赖名称、版本号及本地源码安装记录。

```text
absl-py==2.4.0
albucore==0.0.24
albumentations==2.0.8
annotated-types==0.7.0
certifi==2026.5.20
charset-normalizer==3.4.7
colorama==0.4.6
contourpy==1.3.3
cycler==0.12.1
fast-histogram==0.14
filelock==3.29.0
flatbuffers==25.12.19
fonttools==4.63.0
fsspec==2026.4.0
grpcio==1.80.0
idna==3.16
Jinja2==3.1.6
kiwisolver==1.5.0
Markdown==3.10.2
MarkupSafe==3.0.3
matplotlib==3.10.9
ml_dtypes==0.5.4
mpmath==1.3.0
networkx==3.6.1
numpy==1.26.4
nvidia-cublas-cu12==12.1.3.1
nvidia-cuda-cupti-cu12==12.1.105
nvidia-cuda-nvrtc-cu12==12.1.105
nvidia-cuda-runtime-cu12==12.1.105
nvidia-cudnn-cu12==9.1.0.70
nvidia-cufft-cu12==11.0.2.54
nvidia-curand-cu12==10.3.2.106
nvidia-cusolver-cu12==11.4.5.107
nvidia-cusparse-cu12==12.1.0.106
nvidia-nccl-cu12==2.20.5
nvidia-nvjitlink-cu12==12.4.127
nvidia-nvtx-cu12==12.1.105
onnx==1.18.0
onnxruntime==1.26.0
onnxslim==0.1.94
opencv-python==4.10.0.84
opencv-python-headless==4.10.0.84
packaging==26.0
pillow==12.2.0
polars==1.40.1
polars-runtime-32==1.40.1
protobuf==4.25.4
psutil==7.2.2
pydantic==2.13.4
pydantic_core==2.46.4
pyparsing==3.3.2
python-dateutil==2.9.0.post0
PyYAML==6.0.3
requests==2.34.2
rknn-toolkit2==2.3.2
ruamel.yaml==0.19.1
scipy==1.17.1
simsimd==6.5.16
six==1.17.0
stringzilla==4.6.1
sympy==1.13.1
tensorboard==2.20.0
tensorboard-data-server==0.7.2
torch==2.4.0
torchaudio==2.4.0
torchvision==0.19.0
tqdm==4.67.3
triton==3.0.0
typing-inspection==0.4.2
typing_extensions==4.15.0
# Editable install with no version control (ultralytics==8.4.53)
-e /home/kang/fire1/ultralytics-main
ultralytics-thop==2.0.19
urllib3==2.7.0
Werkzeug==3.1.8
```

注意：editable install 的路径属于本次服务器环境，迁移时需要替换为新机器上的 Ultralytics 源码路径。

## 9. Git 管理说明

COCO2017 数据集约 40 GB，模型权重和部分大型模型文件也不适合直接提交到普通 Git 仓库。

当前 `.gitignore` 会忽略数据集、模型权重、缓存和临时文件。训练参数、CSV 指标记录、脚本、README 和实验曲线等轻量文件应保留在版本管理中。

建议将数据集、预训练权重及训练生成的模型权重单独备份，并在需要迁移时恢复到对应目录。

## 10. 实验总结

本项目完成了一次基于 YOLOv8n 和 COCO2017 的 100 轮目标检测复现实验，保留了训练参数、逐轮训练记录、模型权重、验证指标和可视化结果。

本次实验报告中的指标来自 COCO2017 验证集。由于没有独立测试集，不对独立测试性能作出结论。
