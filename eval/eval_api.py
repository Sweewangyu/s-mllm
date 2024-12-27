# vllm_openai_completions.py
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-xxx", # 随便填写，只是为了通过接口参数校验
)

completion = client.chat.completions.create(
  model="Qwen2-VL-2B-Instruct",
  messages = [
    	{"role": "system", "content": "你是一个有用的助手。"},
    	{"role": "user", "content": [
        {"type": "image_url",
         "image_url": {
           "url": "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"}
        },
        {"type": "text", "text": "插图中的文本是什么？"}
    	]
      }
    ]
)

print(completion.choices[0].message)