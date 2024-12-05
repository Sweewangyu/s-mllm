import logging
from dataclasses import dataclass, field
from transformers import (
    AutoProcessor,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)
from data import LlavaDataset, TrainLLavaModelCollator
from util import *
import torch
logger = logging.getLogger(__name__)

@dataclass
class Arguments:
    model_name_or_path: str = field(default="mllm_chinese")
    train_type: str = field(
        default="freeze_vision_and_llm",
        metadata={"help": "Training types: 'use_lora', 'freeze_vision', 'freeze_vision_and_llm'"}
    )
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})

# 加载模型和处理器
def load_model_and_processor(args: Arguments):
    model = transformers.LlavaForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    processor = transformers.LlavaProcessor.from_pretrained(args.model_name_or_path)

    # 根据训练方式配置模型
    if args.train_type == "use_lora":
        model = setup_lora(model)
    elif args.train_type == "freeze_vision":
        freeze_vision_tower(model)
    elif args.train_type == "freeze_vision_and_llm":
        freeze_vision_and_llm(model)

    print_trainable_parameters(model)
    return model, processor

# 训练过程
import matplotlib.pyplot as plt


# 训练过程
def train():
    parser = HfArgumentParser((Arguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    print(f"Parsed arguments: {args}")

    model, processor = load_model_and_processor(args)
    data_collator = TrainLLavaModelCollator(processor, -100)
    train_dataset = LlavaDataset(args.data_path)

    # 定义一个回调函数来记录损失
    class LossLoggerCallback(transformers.TrainerCallback):
        def __init__(self):
            self.losses = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if 'loss' in logs:
                self.losses.append(logs['loss'])

    # 创建LossLoggerCallback实例
    loss_logger = LossLoggerCallback()

    # 创建Trainer对象
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        callbacks=[loss_logger],  # 添加回调
    )

    # 开始训练
    trainer.train()
    trainer.save_model(output_dir=training_args.output_dir)

    # 绘制损失曲线
    plot_loss_curve(loss_logger.losses)



# 程序入口
if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    train()
