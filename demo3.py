from PIL import Image
import os

# 设置图片路径和输出路径
input_folder = '/home/wangyu/image'  # 替换为您的文件夹路径

# 定义目标尺寸
target_size = (448, 448)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    if filename.lower().endswith('.jpg'):
        # 构建完整的文件路径
        file_path = os.path.join(input_folder, filename)
        try:
            # 打开图像文件
            with Image.open(file_path) as img:
                # 获取图像原始尺寸
                original_size = img.size

                # 检查图像是否已经为目标尺寸
                if original_size != target_size:
                    # 将图像调整为448x448像素
                    resized_img = img.resize(target_size, Image.Resampling.LANCZOS)

                else:

                    resized_img = img  # 如果已经是正确尺寸，则不调整

                # 保存调整大小后的图像到原路径，覆盖原始文件
                resized_img.save(file_path)
                print(f"Overwritten: {file_path}")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

print("All images have been processed.")