#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_PATH="${MODEL_PATH:-train/weights/best.pt}"

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "错误：找不到模型权重：$MODEL_PATH"
    exit 1
fi

# 防止覆盖已有验证结果
if [[ -n "$(find "$ROOT_DIR/val" -maxdepth 1 -type f -print -quit)" ]]; then
    echo "错误：val/ 中已经存在验证结果。"
    echo "请先备份或整理现有结果，再执行独立验证。"
    exit 1
fi

# 使用物理 GPU 2；进程内部对应 device=0
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n yolo311 \
yolo detect val \
model="$MODEL_PATH" \
data=configs/coco.yaml \
imgsz=640 \
batch=16 \
device=0 \
workers=8 \
plots=True \
project="$ROOT_DIR" \
name=val
