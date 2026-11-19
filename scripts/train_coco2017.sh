#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# 防止重新训练时覆盖已有实验结果
if [[ -n "$(find "$ROOT_DIR/train" -maxdepth 1 -type f -print -quit)" || -n "$(find "$ROOT_DIR/train/weights" -maxdepth 1 -type f -print -quit)" ]]; then
    echo "错误：train/ 中已经存在实验结果。"
    echo "请先备份或整理现有结果，再启动新的训练。"
    exit 1
fi

# 使用物理 GPU 2；进程内部对应 device=0
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n yolo311 \
yolo detect train \
model=weights/yolov8n.pt \
data=configs/coco.yaml \
epochs=100 \
imgsz=640 \
batch=16 \
device=0 \
workers=8 \
amp=False \
pretrained=True \
project="$ROOT_DIR" \
name=train \
plots=True \
save=True
