#!/bin/bash

# 设置CUDA可见设备，默认使用第一张卡，可以根据需要调整，例如 CUDA_VISIBLE_DEVICES="0,1"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 设置模型路径和数据路径
MODEL_PATH="${1:-/root/autodl-fs/Qwen2.5-3B-TCM-GRPO-v1}"
DATA_PATH="/root/MedicalGPT/TCM-Text-Exams.json"
OUTPUT_PATH="/root/MedicalGPT/eval_results.json"

echo "开始运行 TCM-Text-Exams 评测..."
echo "使用的模型: ${MODEL_PATH}"
echo "评测数据集: ${DATA_PATH}"

# 运行 Python 评估脚本
python scripts/eval_tcm_benchmark.py \
    --model_path "${MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --output_path "${OUTPUT_PATH}"

echo "评测完成！结果保存在: ${OUTPUT_PATH}"
