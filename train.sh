#!/usr/bin/bash
accelerate launch  --config_file accelerate_one_gpu.yaml run.py \
    --model_name_or_path mllm_chinese \
    --train_type freeze_vision_and_llm \
    --data_path ../ch_llava/pre/image \
    --bf16 true \
    --fp16 false \
    --output_dir output_test_model \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --lr_scheduler_type="cosine" \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --report_to "tensorboard" \
    --learning_rate 1 \
    --logging_steps 5

# 使用方式，后面接一个训练脚本即可， 如：./train.sh sft.py
