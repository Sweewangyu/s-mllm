import torch
from transformers import AutoProcessor
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass
class QaImageOutput:
    q_input_ids: torch.Tensor
    pixel_values: torch.Tensor
    a_input_ids: torch.Tensor

def build_qaimage(processor: AutoProcessor, q_text: str, a_text: str, image_path: Path) -> QaImageOutput:
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": q_text}]
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    try:
        raw_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Error opening image {image_path}: {e}")

    inputs = processor(prompt, raw_image, return_tensors="pt")
    a_input_ids = processor.tokenizer(a_text, return_tensors="pt", padding="longest", truncation=True)["input_ids"]

    return QaImageOutput(q_input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], a_input_ids=a_input_ids)

class LlavaDataset(Dataset):
    def __init__(self, dataset_dir: str) -> None:
        self.chat_data, self.data_dir = self.build_dataset(dataset_dir)
        self.flattened_data = self.flatten_conversations()

    def build_dataset(self, data_dir: str) -> Tuple[List[Dict], Path]:
        data_dir = Path(data_dir)
        chat_file = data_dir.joinpath("chat.json")
        chat_data = pd.read_json(chat_file).to_dict(orient="records")
        return chat_data, data_dir

    def flatten_conversations(self) -> List[Tuple[str, str, Path]]:
        """
        Flatten the conversations into individual question-answer-image samples.
        """
        flattened_data = []
        for item in self.chat_data:
            image_path = self.data_dir.joinpath(item.get("image"))
            conversations = item.get("conversations")

            for i in range(0, len(conversations) - 1, 2):  # Iterate over human-GPT pairs
                if conversations[i].get("from") == "human" and conversations[i + 1].get("from") == "gpt":
                    human_input = conversations[i].get("value")
                    chatbot_output = conversations[i + 1].get("value")

                    # Add <image> token if not present
                    if "<image>" not in human_input:
                        human_input = f"<image>\n{human_input}"

                    flattened_data.append((human_input, chatbot_output, image_path))

        return flattened_data

    def __len__(self):
        return len(self.flattened_data)

    def __getitem__(self, index) -> Tuple[str, str, Path]:
        """
        Return a single (human_input, chatbot_output, image_path) triplet.
        """
        return self.flattened_data[index]



class TrainLLavaModelCollator:
    def __init__(self, processor: AutoProcessor, IGNORE_INDEX: int) -> None:
        self.processor = processor
        self.ignore_index = IGNORE_INDEX

    def convert_one_piece(self, q_input_ids: torch.Tensor, a_input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        eos_token_id_tensor = torch.tensor([self.processor.tokenizer.eos_token_id]).unsqueeze(0)

        # Ensure both tensors have the same number of dimensions
        q_input_ids = q_input_ids.view(1, -1)
        a_input_ids = a_input_ids.view(1, -1)

        input_ids = torch.cat([q_input_ids, a_input_ids, eos_token_id_tensor], dim=1)
        labels = torch.cat([torch.full_like(q_input_ids, self.ignore_index), a_input_ids, eos_token_id_tensor], dim=1)
        return input_ids, labels

    def __call__(self, features: List) -> Dict[str, torch.Tensor]:
        input_ids_list, labels_list, pixel_values, max_input_len_list = [], [], [], []

        for feature in features:
            qaimage_output = build_qaimage(self.processor, feature[0], feature[1], feature[2])
            temp_input_ids, temp_labels = self.convert_one_piece(qaimage_output.q_input_ids, qaimage_output.a_input_ids)

            # Ensure input IDs are of the correct type (Long)
            temp_input_ids = temp_input_ids.long()
            temp_labels = temp_labels.long()

            max_input_len_list.append(temp_input_ids.shape[1])
            input_ids_list.append(temp_input_ids)
            labels_list.append(temp_labels)
            pixel_values.append(qaimage_output.pixel_values)

        max_input_len = max(max_input_len_list)
        final_input_ids, final_labels = [], []

        for index, value in enumerate(input_ids_list):
            padding_length = max_input_len - max_input_len_list[index]
            padded_input_ids = torch.cat([torch.full((1, padding_length), self.processor.tokenizer.pad_token_id), value], dim=1)
            padded_labels = torch.cat([torch.full((1, padding_length), self.ignore_index), labels_list[index]], dim=1)
            final_input_ids.append(padded_input_ids)
            final_labels.append(padded_labels)

        final_input_ids = torch.cat(final_input_ids)
        final_labels = torch.cat(final_labels)
        final_pixel_values = torch.cat(pixel_values, dim=0)
        attention_mask = torch.ones_like(final_input_ids)
        attention_mask[final_input_ids == self.processor.tokenizer.pad_token_id] = 0

        return {
            "input_ids": final_input_ids,
            "labels": final_labels,
            "pixel_values": final_pixel_values,
            "attention_mask": attention_mask
        }

if __name__ == "__main__":
    data_dir ="/home/wangyu/桌面/en_llava/en_pre/images"

    llavadataset = LlavaDataset(data_dir)
    print(len(llavadataset))
    print(llavadataset[0])

