from transformers import LlavaProcessor, LlavaForConditionalGeneration
import torch


model_name_or_path = "show_model/model001"  #

llava_processor = LlavaProcessor.from_pretrained(model_name_or_path)
model = LlavaForConditionalGeneration.from_pretrained(
    model_name_or_path, device_map="cuda:0", torch_dtype=torch.bfloat16
)
print(model)