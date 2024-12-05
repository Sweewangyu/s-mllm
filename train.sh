#!/usr/bin/bash
accelerate launch  --config_file accelerate_one_gpu.yaml run.py \
    --model_name_or_path mllm_chinese_pre \
    --train_type freeze_vision \
    --data_path ../ch_llava/ft/images \
    --bf16 true \
    --output_dir output_test_model \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --report_to "tensorboard" \
    --logging_steps 5

# 使用方式，后面接一个训练脚本即可， 如：./train.sh sft.py
