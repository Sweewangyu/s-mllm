from torch.optim import AdamW
from utils.compute_para import *

from utils.train_type import *
from utils.lr_scheduler import *
from dataclasses import dataclass, field
from transformers import (
    AutoProcessor,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)
from data import LlavaDataset, TrainLLavaModelCollator
logger = logging.getLogger(__name__)
@dataclass
class Arguments:
    model_name_or_path: str = field(default="mllm_chinese")
    train_type: str = field(
        default="freeze_vision_and_llm",
        metadata={"help": "Training types: 'use_lora', 'freeze_vision', 'freeze_vision_and_llm'"}
    )
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})

# 训练过程
def train():
    parser = HfArgumentParser((Arguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    print(f"Parsed arguments: {args}")

    model, processor = load_model_and_processor(args)
    data_collator = TrainLLavaModelCollator(processor, -100)
    train_dataset = LlavaDataset(args.data_path)
    print(model)

    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.1)
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
        callbacks=[loss_logger],
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
