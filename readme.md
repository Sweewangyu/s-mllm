# LLaVA

## 0. 准备

- 一张 3090 显卡
- 百度网盘

### mllm_en：
- 文本部分使用了 `llama3.2-1b-instruct`
- 图像部分使用了 `siglip-so400m-patch14-384`

### mllm_ch：
- 文本部分使用了 `qwen2.5-0.5b-instruct`
- 图像部分使用了 `OFA-Sys/chinese-clip-vit-base-patch16`

将下载好的 `mllm_en` 或 `mllm_ch` 文件夹拖入指定目录，并将 `train.sh` 中的 `model_name_or_path` 修改为对应路径。

---

## 1. 模型结构

![模型结构](https://github.com/Sweewangyu/s-mllm/_resources/80284efbdd653c1f2a4f2fc46005193b.png)

---

## 2. 数据来源

- 英文数据来自 LLaVA 的官方数据集，地址如下：  
  [https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)

- 中文数据集包括：
  - CogVLM-SFT-311K 数据集  
    [https://github.com/THUDM/CogVLM/blob/main/dataset_zh.md](https://github.com/THUDM/CogVLM/blob/main/dataset_zh.md)
  - 自制数据集：基于通义 SA1B 开源数据（转换为对话形式，目前完成 100 万条）  
    [https://modelscope.cn/datasets/Tongyi-DataEngine/SA1B-Dense-Caption](https://modelscope.cn/datasets/Tongyi-DataEngine/SA1B-Dense-Caption)

---

## 3. 训练过程

### 阶段一：特征对齐预训练

由于从 CLIP 提取的特征与 word embedding 不在同一个语义表达空间，因此需要通过预训练，将 image token embedding 对齐到 text word embedding 的语义表达空间。  
- 冻结 Vision Encoder 和 LLM 模型的权重参数，仅训练插值层 `Projection W` 的权重。  
- 修改 `tran.sh` 中的 `train_type` 为 `freeze_vision_and_llm`。

#### mllm_en 预训练的损失函数：
![mllm_en_loss](https://github.com/Sweewangyu/s-mllm/_resources/training_loss_curve.png)

#### mllm_ch 预训练的损失函数：
![mllm_ch_loss](https://github.com/Sweewangyu/s-mllm/_resources/training_loss_curve-1.png)

---

### 阶段二：端到端训练

- 冻结 Vision Encoder 的权重，同时更新插值层 `Projection W` 和 LLM 语言模型的权重。
- 训练过程中考虑 Multimodal Chatbot 和 Science QA 两种典型任务。  
- 修改 `tran.sh` 中的 `train_type` 为 `freeze_vision`。

#### mllm_en 微调的损失函数：
![mllm_en_tuning_loss](https://github.com/Sweewangyu/s-mllm/_resources/training_loss_curve-3.png)

#### mllm_ch 微调的损失函数：
![mllm_ch_tuning_loss](https://github.com/Sweewangyu/s-mllm/_resources/training_loss_curve-2.png)

---

## 测试

![测试图片](https://github.com/Sweewangyu/s-mllm/_resources/1.jpg)

### mllm_en 的回答：
> The image features a cluster of white flowers, specifically daisies, growing in a field.

### mllm_ch 的回答：
> 一朵白色的雏菊，花瓣展开，中心有黄色的花蕊。

---

## 4. Eval

- 部署本地语言模型作为评判 / 选择提取器：
  ```bash
  git clone https://github.com/open-compass/VLMEvalKit.git
  cd VLMEvalKit
  pip install -e .
  pip install lmdeploy openai
