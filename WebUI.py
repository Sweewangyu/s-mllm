import gradio as gr
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration


# 加载模型和处理器
def load_model(model_name_or_path: str):
    model = LlavaForConditionalGeneration.from_pretrained(model_name_or_path)
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    return model, processor


# 定义模型推理函数
def inference(text_input, image_input):
    # 假设模型支持图像和文本输入
    inputs = processor(text=text_input, images=image_input, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.generate(**inputs)

    # 假设模型输出是生成的文本
    generated_text = processor.decode(outputs[0], skip_special_tokens=True)
    return generated_text


# 设置Gradio界面
def create_gradio_interface():
    text_input = gr.Textbox(label="Text Input", placeholder="Enter text here...")
    image_input = gr.Image(type="pil", label="Image Input", source="upload")
    output = gr.Textbox(label="Generated Text")

    # 创建Gradio界面
    gr.Interface(fn=inference, inputs=[text_input, image_input], outputs=output).launch()


if __name__ == "__main__":
    model_name_or_path = "your_model_path_or_name_here"  # 替换为你的模型路径或名称
    model, processor = load_model(model_name_or_path)

    create_gradio_interface()
