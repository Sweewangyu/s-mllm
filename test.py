from transformers import LlavaProcessor, LlavaForConditionalGeneration
import torch


model_name_or_path = "show_model/model001"  #
# model_name_or_path = "test_model_copy/model001"  #

llava_processor = LlavaProcessor.from_pretrained(model_name_or_path)
model = LlavaForConditionalGeneration.from_pretrained(
    model_name_or_path, device_map="cuda:0", torch_dtype=torch.bfloat16
)

from PIL import Image

prompt_text = "<image>\n这是什么?"


messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt_text},
]
prompt = llava_processor.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)


image_path = "000000039769.jpg"
image = Image.open(image_path)


inputs = llava_processor(text=prompt, images=image, return_tensors="pt")

for tk in inputs.keys():
    inputs[tk] = inputs[tk].to(model.device)
generate_ids = model.generate(**inputs, max_new_tokens=20)
gen_text = llava_processor.batch_decode(
    generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
)[0]

print(gen_text)