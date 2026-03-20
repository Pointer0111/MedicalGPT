#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/src
python merge_peft_adapter.py \
    --base_model /gz-fs/qwen-2.5-3b-sft \
    --lora_model ./outputs-dpo-qwen-v1 \
    --output_dir /gz-fs/qwen-2.5-3b-medical-final
