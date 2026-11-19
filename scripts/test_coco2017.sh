#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_PATH="${MODEL_PATH:-train/weights/best.pt}"
TEST_DATA_YAML="${TEST_DATA_YAML:-}"

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "错误：找不到模型权重：$MODEL_PATH"
    exit 1
fi

if [[ -z "$TEST_DATA_YAML" ]]; then
    echo "错误：尚未指定独立测试集配置。"
    echo "请设置 TEST_DATA_YAML=/path/to/test.yaml 后再运行。"
    echo "注意：当前项目只有 train2017 和 val2017，不能将 val2017 冒充独立测试集。"
    exit 1
fi

if [[ ! -f "$TEST_DATA_YAML" ]]; then
    echo "错误：找不到测试集配置：$TEST_DATA_YAML"
    exit 1
fi

if [[ -n "$(find "$ROOT_DIR/test" -maxdepth 1 -type f -print -quit)" ]]; then
    echo "错误：test/ 中已经存在测试结果，请先备份。"
    exit 1
fi

# 使用物理 GPU 2；进程内部对应 device=0
CUDA_VISIBLE_DEVICES=2 conda run --no-capture-output -n yolo311 \
yolo detect val \
model="$MODEL_PATH" \
data="$TEST_DATA_YAML" \
imgsz=640 \
batch=16 \
device=0 \
workers=8 \
plots=True \
project="$ROOT_DIR" \
name=test
