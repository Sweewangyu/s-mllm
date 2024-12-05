import torch.nn as nn
import logging
import matplotlib.pyplot as plt
import transformers
from peft import LoraConfig, get_peft_model
from transformers import get_wsd_schedule
logger = logging.getLogger(__name__)
def get_nb_trainable_parameters(model:nn.Module) -> tuple[int, int]:
    r"""
    Returns the number of trainable parameters and the number of all parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            if hasattr(param, "element_size"):
                num_bytes = param.element_size()
            elif not hasattr(param, "quant_storage"):
                num_bytes = 1
            else:
                num_bytes = param.quant_storage.itemsize
            num_params = num_params * 2 * num_bytes

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def print_trainable_parameters(model: nn.Module) -> None:
    """
    Prints the number of trainable parameters in the model.

    Note: print_trainable_parameters() uses get_nb_trainable_parameters() which is different from
    num_parameters(only_trainable=True) from huggingface/transformers. get_nb_trainable_parameters() returns
    (trainable parameters, all parameters) of the Peft Model which includes modified backbone transformer model.
    For techniques like LoRA, the backbone transformer model is modified in place with LoRA modules. However, for
    prompt tuning, the backbone transformer model is unmodified. num_parameters(only_trainable=True) returns number
    of trainable parameters of the backbone transformer model which can be different.
    """
    trainable_params, all_param = get_nb_trainable_parameters(model)

    print(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
    )

# Lora配置
def setup_lora(model):
    logger.warning("Loading model with LoRA configuration")
    config = LoraConfig(
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["multi_modal_projector"],
    )
    return get_peft_model(model, config)


# 冻结vision_tower层
def freeze_vision_tower(model):
    logger.warning("Freezing vision_tower layers")
    for param in model.vision_tower.parameters():
        param.requires_grad = False


# 冻结vision_tower和llm层，仅训练投影层
def freeze_vision_and_llm(model):
    logger.warning("Freezing vision_tower and LLM layers, only training projector")
    for param in model.vision_tower.parameters():
        param.requires_grad = False
    for param in model.language_model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        param.requires_grad = "multi_modal_projector" in name

def plot_loss_curve(losses, output_dir):
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()

    # 保存图像到指定目录
    loss_plot_path = f"{output_dir}/training_loss_curve.png"
    plt.savefig(loss_plot_path)
    plt.close()  # 关闭图像，释放内存
    print(f"Loss curve saved to: {loss_plot_path}")

def compute_total_steps(train_dataset, training_args):
    total_samples = len(train_dataset)  # 数据集样本总数
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    )
    total_steps = (total_samples // effective_batch_size) * training_args.num_train_epochs
    return total_steps

# 动态计算学习率调度器参数
def custom_lr_scheduler(optimizer, total_steps):
    num_warmup_steps = int(total_steps * 0.05)   # 预热步数为总步数的 5%
    num_stable_steps = int(total_steps * 0.3)   # 恒定阶段为总步数的 30%
    num_decay_steps = int(total_steps * 0.6)    # 衰减阶段为总步数的 60%
    print(f"Warmup Steps: {num_warmup_steps}, Stable Steps: {num_stable_steps}, Decay Steps: {num_decay_steps}")
    return get_wsd_schedule(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_stable_steps=num_stable_steps,
        num_decay_steps=num_decay_steps,
        min_lr_ratio=0.1,  # 最小学习率比例
        num_cycles=0.5     # 半周期余弦
    )