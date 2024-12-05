# 训练llava
ch_mmlm
1. 模型构建：基于`OFA-Sys/chinese-clip-vit-base-patch16` 和`Qwen2.5-0.5B-Chat`模型，构建一个ch_llava模型
2. 数据构建：`cogvlm2`
3. 训练方式：基于`deepspeed-zero2`，有`lora`训练、全量参数训练、冻结视觉层进行训练等方式。

en_mllm
1. 模型构建：基于`OFA-Sys/chinese-clip-vit-base-patch16` 和`Llama3.2-1B-Instruct`模型，构建一个en_llava模型
2. 数据构建：`LLaVA-CC3M-Pretrain-595K`
3. 训练方式：基于`deepspeed-zero2`，有`lora`训练、全量参数训练、冻结视觉层进行训练等方式。
## 具体教程


