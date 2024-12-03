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

logger = logging.getLogger(__name__)

@dataclass
class Arguments:
    model_name_or_path: str = field(default="mllm")
    train_type: str = field(
        default="use_lora",
        metadata={
            "help": "Training types: 'use_lora', 'freeze_vision', 'freeze_vision_and_llm'"
        },
    )
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    output_dir: str = field(default="./output", metadata={"help": "Where to save the model."})
    num_train_epochs: int = field(default=3, metadata={"help": "Number of training epochs."})
    per_device_train_batch_size: int = field(default=8, metadata={"help": "Batch size per device."})
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates."})
    logging_dir: str = field(default="./logs", metadata={"help": "Directory for logs."})
    learning_rate: float = field(default=5e-5, metadata={"help": "Learning rate for training."})

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
def train():
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args_into_dataclasses()[0]

    model, processor = load_model_and_processor(args)
    data_collator = TrainLLavaModelCollator(processor, -100)
    train_dataset = LlavaDataset(args.data_path)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        save_steps=args.save_steps,
        logging_dir=args.logging_dir,
        learning_rate=args.learning_rate,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(output_dir=args.output_dir)

# 程序入口
if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    train()
