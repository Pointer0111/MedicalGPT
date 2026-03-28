#!/bin/bash
set -euo pipefail


if [ ! -f "/root/medical/grpo_tcm/train.jsonl" ]; then
    echo "未找到训练数据: /root/medical/grpo_tcm/train.jsonl"
    echo "请先执行: bash scripts/run_convert_grpo_tcm.sh"
    exit 1
fi

# 使用 SFT 合并后的中医模型进行 GRPO 训练
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-384}"

if ! python -c "import weave" >/dev/null 2>&1; then
    echo "缺少依赖: weave"
    echo "请先执行: pip install weave"
    exit 1
fi

torchrun --nproc_per_node "${NPROC_PER_NODE}" training/grpo_training.py \
    --model_name_or_path /root/autodl-fs/Qwen2.5-3B-TCM-SFT \
    --train_file_dir /root/medical/grpo_tcm \
    --train_samples 2000 \
    --max_steps -1 --num_train_epochs 1 \
    --save_steps 50 \
    --save_strategy steps \
    --save_total_limit 3 \
    --output_dir outputs-grpo-tcm-v2 \
    --dtype bfloat16 \
    --bf16 True \
    --report_to tensorboard \
    --remove_unused_columns False \
    --gradient_checkpointing True \
    --beta 0.001 \
    --learning_rate 5.0e-7 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --use_vllm False \
    --logging_steps 10 \
    --use_peft True \
    --qlora False \
    --load_in_4bit False \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --num_generations "${NUM_GENERATIONS}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --max_completion_length "${MAX_COMPLETION_LENGTH}"

echo "GRPO 训练完成!"
