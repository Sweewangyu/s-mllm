import os
import json
import os
import json


def filter_chat_json(chat_json_path, images_base_path, output_json_path):
    """
    过滤掉缺失图片的条目，并保存新的 JSON 文件。

    Args:
        chat_json_path (str): 原始 `chat.json` 文件路径。
        images_base_path (str): 图片所在的根目录路径。
        output_json_path (str): 保存过滤后 JSON 文件的路径。
    """
    with open(chat_json_path, "r") as f:
        chat_data = json.load(f)

    filtered_data = []
    for entry in chat_data:
        image_path = os.path.join(images_base_path, entry["image"])
        if os.path.exists(image_path):
            filtered_data.append(entry)
        else:
            print(f"Missing image: {image_path}, skipping entry ID: {entry['id']}")

    # 将过滤后的数据保存到新文件
    with open(output_json_path, "w") as f:
        json.dump(filtered_data, f, indent=2)

    print(f"Filtered data saved to {output_json_path}. Original: {len(chat_data)}, Filtered: {len(filtered_data)}")


# 示例调用
chat_json_path = "../en_llava/en_pre/images/chat.json"  # 原始 chat.json 文件路径
images_base_path = "../en_llava/en_pre/images"  # 图片根目录路径
output_json_path = "../en_llava/filtered_chat.json"  # 保存的过滤后文件路径

filter_chat_json(chat_json_path, images_base_path, output_json_path)

