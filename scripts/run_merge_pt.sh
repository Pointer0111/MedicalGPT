#!/bin/bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd):$(pwd)/src"

BASE_MODEL="/gz-fs/Qwen2.5-3B-TCM-SFT"
LORA_MODEL="./outputs-grpo-tcm-v2"
OUTPUT_DIR="/gz-fs/Qwen2.5-3B-TCM-GRPO"

if [ ! -d "$BASE_MODEL" ]; then
    echo "未找到基础模型目录: $BASE_MODEL"
    exit 1
fi

if [ ! -d "$LORA_MODEL" ]; then
    echo "未找到 LoRA 目录: $LORA_MODEL"
    echo "请先执行 GRPO 训练: bash scripts/run_grpo_tcm.sh"
    exit 1
fi

python tools/merge_peft_adapter.py \
    --base_model "$BASE_MODEL" \
    --lora_model "$LORA_MODEL" \
    --output_dir "$OUTPUT_DIR"

echo "合并完成: $OUTPUT_DIR"
