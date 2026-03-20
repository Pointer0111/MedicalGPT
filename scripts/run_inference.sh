#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/src
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --base_model /gz-fs/Qwen2.5-3B \
    --lora_model ./outputs-pt-qwen-v1 \
    --interactive \
    --temperature 0.7 \
    --max_new_tokens 512
