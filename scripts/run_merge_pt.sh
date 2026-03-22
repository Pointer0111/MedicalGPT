#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/src

# 合并SFT阶段的 LoRA 适配器到 PT 模型中
python tools/merge_peft_adapter.py \
    --base_model /gz-fs/Qwen2.5-3B-TCM-PT \
    --lora_model ./outputs-sft-qwen-v1 \
    --output_dir /gz-fs/Qwen2.5-3B-TCM-SFT
