LLaVA
0.准备
一张3090显卡

百度网盘
mll_en：
text部分用了llama3.2-1b-instruct
image部分用了siglip-so400m-patch14-384

mllm_ch
text部分用了qwen2.5-0.5b-instruct
image部分用了OFA-Sys/chinese-clip-vit-base-patch16

将下载好的mllm_en或者mllm_ch拖入文件夹中
把train.sh下的model_name_or_path，改成该路径

1.模型结构
![80284efbdd653c1f2a4f2fc46005193b.png](../_resources/80284efbdd653c1f2a4f2fc46005193b.png)
flash_attention



2.数据来源
英文数据来自llava的官方数据集，地址如下:
https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain

中文数据集除了CogVLM-SFT-311K:
https://github.com/THUDM/CogVLM/blob/main/dataset_zh.md

自己制作了来自通义的SA1B发布的开源数据集(修改成对话的形式，目前只制作了100w):
https://modelscope.cn/datasets/Tongyi-DataEngine/SA1B-Dense-Caption


3.训练过程
阶段一：特征对齐预训练。由于从CLIP提取的特征与word embedding不在同一个语义表达空间，因此，需要通过预训练，将image token embedding对齐到text word embedding的语义表达空间。这个阶段冻结Vision Encoder和LLM模型的权重参数，只训练插值层Projection W的权重
将tran.sh中的 train_type设为freeze_vision_and_llm

mllm_en预训练的损失函数：
![training_loss_curve.png](../_resources/training_loss_curve.png)
mllm_ch预训练的损失函数：
![training_loss_curve.png](../_resources/training_loss_curve-1.png)

阶段二：端到端训练。这个阶段，依然冻结Vision Encoder的权重，训练过程中同时更新插值层Projection W和LLM语言模型的权重，训练考虑Multimodal Chatbot和Science QA两种典型的任务
将tran.sh中的 train_type设为freeze_vision

mllm_en微调的损失函数：
![training_loss_curve.png](../_resources/training_loss_curve-3.png)
mllm_ch微调的损失函数：
![training_loss_curve.png](../_resources/training_loss_curve-2.png)

测试
![1.jpg](../_resources/1.jpg)
这是mllm_en回答：
The image features a cluster of white flowers, specifically daisies, growing in a field
这是mllm_ch回答
一朵白色的雏菊，花瓣展开，中心有黄色的花蕊


4.eval
部署本地语言模型作为评判 / 选择提取器
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
pip install -e .
pip install lmdeploy openai

VLMEvalKit目前存在问题，还在努力中