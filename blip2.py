from transformers import (
    Blip2VisionConfig,
    Blip2QFormerConfig,
    OPTConfig,
    Blip2Config,
    Blip2ForConditionalGeneration,
)
from transformers import AutoTokenizer, AutoModelForCausalLM

qwen_model_name_or_path = r"E:\huggingface\cache\models--Qwen--Qwen2-0.5B-Instruct\snapshots\c291d6fce4804a1d39305f388dd32897d1f7acc4"
llm_model = AutoModelForCausalLM.from_pretrained(qwen_model_name_or_path, device_map="cuda:0")
vision_config = Blip2VisionConfig()
qformer_config = Blip2QFormerConfig()
text_config = llm_model.config

config = Blip2Config.from_vision_qformer_text_configs(vision_config, qformer_config, text_config)
model = Blip2ForConditionalGeneration(config)
model.save_pretrained("show_model/blip2-qwen2")
print(model)