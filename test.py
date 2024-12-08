from transformers import LlavaProcessor, LlavaForConditionalGeneration
import torch
from PIL import Image
model_name_or_path = "/home/wangyu/桌面/mllm权重/mllm_en_ft/checkpoint-8509"  #
llava_processor = LlavaProcessor.from_pretrained(model_name_or_path)
model = LlavaForConditionalGeneration.from_pretrained(model_name_or_path, device_map="cuda:0")

prompt_text = "<image>\nwhat is it"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt_text},
]

prompt = llava_processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


image_path = "1.jpg"
image = Image.open(image_path)
inputs = llava_processor(text=prompt, images=image, return_tensors="pt")

for tk in inputs.keys():
    inputs[tk] = inputs[tk].to(model.device)
generate_ids = model.generate(**inputs, max_new_tokens=20)
gen_text = llava_processor.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

print(gen_text)