#!/usr/bin/bash
accelerate launch  --config_file accelerate_one_gpu.yaml run.py \
    --model_name_or_path show_model/model001 \
    --train_type use_lora \
    --data_path ../data \
    --bf16 true \
    --fp16 false \
    --output_dir output_test_model \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --report_to "tensorboard" \
    --learning_rate 4e-4 \
    --logging_steps 10 

# 使用方式，后面接一个训练脚本即可， 如：./train.sh sft.py
