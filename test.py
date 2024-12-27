from transformers import LlavaProcessor, LlavaForConditionalGeneration
import torch
from PIL import Image
model_name_or_path = "/home/wangyu/桌面/mllm权重/mllm_ch/mllm_ch_sft"  #
llava_processor = LlavaProcessor.from_pretrained(model_name_or_path,
                                                torch_dtype=torch.float16,
                                                device_map="cuda:0",
                                                 #attn_implementation="flash_attention_2"
                                                 )
model = LlavaForConditionalGeneration.from_pretrained(model_name_or_path,
                                                      device_map="cuda:0",
                                                      torch_dtype=torch.float16,
                                                      #attn_implementation="flash_attention_2"
                                                      )
#prompt_text = "<image>\nwhat is it"
prompt_text = "<image>\n这是什么"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt_text},
]

prompt = llava_processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


image_path = "1.jpg"
image = Image.open(image_path)
inputs = llava_processor(text=prompt, images=image, return_tensors="pt")

for tk in inputs.keys():
    if tk == "attention_mask":  # 确保 attention_mask 是整数类型
        inputs[tk] = inputs[tk].to(model.device, dtype=torch.int64)
    elif tk != "input_ids":  # 其他张量根据模型需求转换为 float16
        inputs[tk] = inputs[tk].to(model.device, dtype=torch.float16)
    else:  # input_ids 保持为 Long 类型
        inputs[tk] = inputs[tk].to(model.device)


generate_ids = model.generate(**inputs, max_new_tokens=20)
gen_text = llava_processor.batch_decode(generate_ids)[0]

print(gen_text)