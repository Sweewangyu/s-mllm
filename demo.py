import torch
from accelerate.commands.config.config_args import cache_dir
from transformers import pipeline

model_id = "unsloth/Llama-3.2-1B-Instruct"
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(model_id,cache_dir='llm')
model = AutoModelForCausalLM.from_pretrained(model_id,cache_dir='llm')
print(tokenizer.encode("image"))

from transformers import pipeline
from PIL import Image
import requests

from PIL import Image
import requests
# from transformers import AutoProcessor, AutoModel
# import torch
#
# model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384",cache_dir='lvm')
# processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384",cache_dir='lvm')


