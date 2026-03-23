#!/bin/bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd):$(pwd)/src"

DATA_FILE="/root/medical/grpo_tcm/valid.jsonl"
SFT_MODEL="/root/autodl-fs/Qwen2.5-3B-TCM-SFT"
GRPO_MODEL="/root/autodl-fs/Qwen2.5-3B-TCM-GRPO"
OUTPUT_DIR="./outputs-grpo-eval"

if [ ! -f "$DATA_FILE" ]; then
    echo "未找到评测数据: $DATA_FILE"
    exit 1
fi

if [ ! -d "$SFT_MODEL" ]; then
    echo "未找到 SFT 模型目录: $SFT_MODEL"
    exit 1
fi

if [ ! -d "$GRPO_MODEL" ]; then
    echo "未找到 GRPO 模型目录: $GRPO_MODEL"
    echo "请先执行合并: bash scripts/run_merge_pt.sh"
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 python tools/eval_grpo_sample.py \
    --data_file "$DATA_FILE" \
    --sft_model "$SFT_MODEL" \
    --grpo_model "$GRPO_MODEL" \
    --sample_size 100 \
    --seed 42 \
    --max_new_tokens 256 \
    --temperature 0.0 \
    --output_dir "$OUTPUT_DIR"
