import logging
from dataclasses import dataclass, field
from transformers import (
    LlavaForConditionalGeneration,
    LlavaProcessor,
    Trainer,
    TrainingArguments,
)
import torch
import transformers
from data import LlavaDataset, TrainLLavaModelCollator
from util import print_trainable_parameters
from peft import LoraConfig, get_peft_model

# 设置日志
logger = logging.getLogger(__name__)

# 定义命令行参数类
@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="test_model/model001")
    train_type: str = field(
        default="use_lora",
        metadata={
            "help": """
            1. use_lora: 使用Lora训练,
            2. freeze_vision: 冻结vision_tower层, 其余层可训练;
            3. freeze_vision_and_llm: 冻结vision_tower和llm，仅训练投影层
            """
        },
    )

@dataclass
class DataArguments:
    data_path: str = field(metadata={"help": "训练数据路径"})

# 加载模型和处理器
def load_model_and_processor(model_args: ModelArguments):
    model = LlavaForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    processor = LlavaProcessor.from_pretrained(model_args.model_name_or_path)

    # 根据不同训练模式进行模型调整
    if model_args.train_type == "use_lora":
        model = apply_lora(model)
    elif model_args.train_type == "freeze_vision":
        freeze_vision(model)
    elif model_args.train_type == "freeze_vision_and_llm":
        freeze_vision_and_llm(model)

    print_trainable_parameters(model)
    return model, processor

# 使用Lora对模型进行调整
def apply_lora(model):
    logger.warning("应用Lora训练模式")
    config = LoraConfig(
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["multi_modal_projector"],
    )
    return get_peft_model(model, config)

# 冻结vision_tower层的参数
def freeze_vision(model):
    logger.warning("冻结vision_tower层，其他层可训练")
    for param in model.vision_tower.parameters():
        param.requires_grad = False

# 冻结vision_tower和llm层的参数，只训练投影层
def freeze_vision_and_llm(model):
    logger.warning("冻结vision_tower和llm层，仅训练投影层")
    for param in model.vision_tower.parameters():
        param.requires_grad = False
    for param in model.language_model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        param.requires_grad = "multi_modal_projector" in name

# 加载训练数据集
def load_train_dataset(data_args: DataArguments):
    return LlavaDataset(data_args.data_path)

# 训练模型
def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model, processor = load_model_and_processor(model_args)
    data_collator = TrainLLavaModelCollator(processor, -100)
    train_dataset = load_train_dataset(data_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

# 主程序入口
if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    train()
