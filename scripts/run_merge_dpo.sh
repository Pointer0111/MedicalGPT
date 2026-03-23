#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/src
python merge_peft_adapter.py \
    --base_model /root/autodl-fs/qwen-2.5-3b-sft \
    --lora_model ./outputs-dpo-qwen-v1 \
    --output_dir /root/autodl-fs/qwen-2.5-3b-medical-final
