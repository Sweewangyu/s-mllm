import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from compassrank import CompassRank
from datasets import load_dataset


# 加载模型和处理器
def load_model(model_name_or_path: str):
    model = LlavaForConditionalGeneration.from_pretrained(model_name_or_path)
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    return model, processor


# 定义推理函数
def inference(text_input, image_input):
    # 假设模型支持图像和文本输入
    inputs = processor(text=text_input, images=image_input, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.generate(**inputs)

    # 假设模型输出是生成的文本
    generated_text = processor.decode(outputs[0], skip_special_tokens=True)
    return generated_text


# 加载数据集 (以"VECO"为例，你可以换成任何支持的中文数据集)
def load_data(dataset_name):
    dataset = load_dataset(dataset_name)
    return dataset


# 使用 CompassRank 评估模型
def evaluate_with_compassrank(model, processor, dataset_name):
    # 加载指定的数据集
    dataset = load_data(dataset_name)

    # 初始化 CompassRank
    ranker = CompassRank(model, processor)

    # 评估模型在数据集上的表现
    results = ranker.evaluate(dataset)

    return results


# 主函数
if __name__ == "__main__":
    model_name_or_path = "your_model_path_or_name_here"  # 替换为你的模型路径或名称
    model, processor = load_model(model_name_or_path)

    # 指定要评估的数据集
    dataset_name = "veco"  # 这里可以替换为其它支持的中文多模态数据集

    # 评估模型
    results = evaluate_with_compassrank(model, processor, dataset_name)

    # 输出评估结果
    print(f"Evaluation Results for {dataset_name}: {results}")
