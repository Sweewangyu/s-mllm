import gradio as gr
from transformers import LlavaForConditionalGeneration,LlavaProcessor
from PIL import Image
import torch
# 加载模型和处理器
def load_model(model_name_or_path: str):
    model = LlavaForConditionalGeneration.from_pretrained(model_name_or_path)
    processor = LlavaProcessor.from_pretrained(model_name_or_path)
    return model, processor

def inference(text_input, image_input):
    if image_input is None:
        return "Please upload an image."
    global model, processor
    text_input += '<image>\n'
    # 处理输入
    image = Image.open(image_input)
    inputs = processor(text=text_input, images=image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    # 推理
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20)
    # 解码生成文本
    generated_text = processor.decode(outputs[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return generated_text


# 创建 Gradio 界面
def create_gradio_interface():
    text_input = gr.Textbox(label="Text Input", placeholder="Enter text here...")
    image_input = gr.Image(type="filepath", label="Image Input")
    output = gr.Textbox(label="Generated Text")


    gr.Interface(
        fn=inference,
        inputs=[text_input, image_input],
        outputs=output,
        title="LLaVA Multimodal Model Demo",
        description="Upload an image and provide a text input to generate responses."
    ).launch()


if __name__ == "__main__":
    model_name_or_path = "/home/wangyu/桌面/smllm/mllm_en_ft/checkpoint-8509"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_model(model_name_or_path)
    model = model.to(device)

    # 启动 Gradio 界面
    create_gradio_interface()
