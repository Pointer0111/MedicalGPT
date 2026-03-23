#!/bin/bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd):$(pwd)/src"

source /etc/network_turbo
rm -rf /root/medical/grpo_tcm
mkdir -p /root/medical/grpo_tcm

python tools/convert_hf_sft_to_grpo.py \
    --dataset_name SylvanL/Traditional-Chinese-Medicine-Dataset-SFT \
    --output_dir /root/medical/grpo_tcm
