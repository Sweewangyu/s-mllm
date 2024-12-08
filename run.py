import logging
import torch
import transformers
from torch.optim import AdamW
# from utils.compute_para import *
# from utils.train_type import *
from dataclasses import dataclass, field
from transformers import (
    AutoProcessor,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)
from data import LlavaDataset, TrainLLavaModelCollator
import flash_attn
from util import *
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
    # 加载模型时指定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = transformers.LlavaForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cuda:0",
        attn_implementation="flash_attention_2"
    ).to(device)  # 立即移动到指定设备

    processor = transformers.LlavaProcessor.from_pretrained(args.model_name_or_path)

    # 根据训练方式配置模型
    if args.train_type == "use_lora":
        model = setup_lora(model) # 确保设置 LoRA 后仍在 GPU 上
    elif args.train_type == "freeze_vision":
        freeze_vision_tower(model) # 确保冻结视觉塔后仍在 GPU 上
    elif args.train_type == "freeze_vision_and_llm":
        freeze_vision_and_llm(model) # 确保冻结视觉和 LLM 后仍在 GPU 上

    print_trainable_parameters(model)
    return model, processor
# 训练过程
def train():
    parser = HfArgumentParser((Arguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    print(f"Parsed arguments: {args}")

    model, processor = load_model_and_processor(args)
    data_collator = TrainLLavaModelCollator(processor, -100)
    train_dataset = LlavaDataset(args.data_path)
    print(model)

    optimizer = AdamW(model.parameters(), lr=4e-5, weight_decay=0.1)
    total_steps = compute_total_steps(train_dataset, training_args)
    # lr_scheduler = custom_lr_scheduler(optimizer, total_steps)
    lr_scheduler = cos_lr_scheduler(optimizer, total_steps)


    # 创建LossLoggerCallback实例
    loss_logger = LossLoggerCallback()

    # 创建Trainer对象
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        optimizers=(optimizer, lr_scheduler),
        eval_dataset=None,
        data_collator=data_collator,
        callbacks=[loss_logger],  # 添加回调
    )

    # 开始训练
    trainer.train()
    trainer.save_model(output_dir=training_args.output_dir)

    # 绘制损失曲线
    plot_loss_curve(loss_logger.losses,output_dir=training_args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    train()
