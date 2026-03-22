#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/src

source /etc/network_turbo
python tools/convert_hf_sft_to_grpo.py \
    --dataset_name SylvanL/Traditional-Chinese-Medicine-Dataset-SFT \
    --output_dir /root/medical/grpo_tcm \
    --validation_ratio 0.01
