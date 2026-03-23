#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/src

source /etc/network_turbo
if [ ! -f /root/medical/finetune_shennong_sharegpt/train.jsonl ]; then
    python tools/convert_dataset.py \
        --dataset_name michaelwzhu/ShenNong_TCM_Dataset \
        --output_dir /root/medical/finetune_shennong_sharegpt \
        --data_type auto \
        --validation_ratio 0.01
fi

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 training/supervised_finetuning.py \
    --model_name_or_path /root/autodl-fs/Qwen2.5-3B-TCM-PT \
    --train_file_dir /root/medical/finetune_shennong_sharegpt \
    --validation_file_dir /root/medical/finetune_shennong_sharegpt \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --use_peft True \
    --seed 42 \
    --max_train_samples 10000 \
    --max_eval_samples 10 \
    --num_train_epochs 1 \
    --learning_rate 2e-4 \
    --warmup_steps 100 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 100 \
    --eval_strategy steps \
    --save_steps 500 \
    --save_strategy steps \
    --save_total_limit 3 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 10 \
    --model_max_length 1024 \
    --output_dir outputs-sft-qwen-v1 \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --torch_dtype bfloat16 \
    --bf16 \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing True \
    --cache_dir ./cache \
    --template_name qwen
