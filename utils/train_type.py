import torch
from transformers import LlavaForConditionalGeneration,LlavaProcessor
from utils.compute_para import *
from peft import LoraConfig, get_peft_model
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)
def load_model_and_processor(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cuda:0",
        #attn_implementation="flash_attention_2"
    ).to(device)  # 立即移动到指定设备

    processor = LlavaProcessor.from_pretrained(args.model_name_or_path,
                                                torch_dtype=torch.bfloat16,
                                                device_map="cuda:0",
                                                # attn_implementation="flash_attention_2"
    )

    # 根据训练方式配置模型
    if args.train_type == "use_lora":
        model = setup_lora(model) # 确保设置 LoRA 后仍在 GPU 上
    elif args.train_type == "freeze_vision":
        freeze_vision_tower(model) # 确保冻结视觉塔后仍在 GPU 上
    elif args.train_type == "freeze_vision_and_llm":
        freeze_vision_and_llm(model) # 确保冻结视觉和 LLM 后仍在 GPU 上

    print_trainable_parameters(model)
    return model, processor
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