#!/usr/bin/bash
accelerate launch  --config_file accelerate_one_gpu.yaml run.py \
    --model_name_or_path mllm_pre \
    --train_type freeze_vision \
    --data_path ../en_llava/en_ft \
    --bf16 true \
    --output_dir mllm_en_ft \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --report_to "tensorboard" \
    --logging_steps 5

# 使用方式，后面接一个训练脚本即可， 如：./train.sh sft.py
