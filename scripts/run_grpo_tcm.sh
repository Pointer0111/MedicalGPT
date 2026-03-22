#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/src

# 使用 SFT 合并后的中医模型进行 GRPO 训练
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 training/grpo_training.py \
    --model_name_or_path /gz-fs/Qwen2.5-3B-TCM-SFT \
    --train_file_dir data/grop \
    --train_samples -1 \
    --max_steps -1 --num_train_epochs 1 \
    --save_steps 50 \
    --save_strategy steps \
    --save_total_limit 3 \
    --output_dir outputs-grpo-tcm-v1 \
    --torch_dtype bfloat16 \
    --bf16 True \
    --report_to tensorboard \
    --remove_unused_columns False \
    --gradient_checkpointing False \
    --beta 0.001 \
    --learning_rate 5.0e-7 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --use_vllm False \
    --logging_steps 10 \
    \
    `# QLoRA配置` \
    --use_peft True \
    --qlora True \
    --load_in_4bit True \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    \
    `# 显存优化配置` \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --num_generations 4 \
    --gradient_accumulation_steps 2 \
    --max_prompt_length 1024 \
    --max_completion_length 512

echo "GRPO 训练完成!"
