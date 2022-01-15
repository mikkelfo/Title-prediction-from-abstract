from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import BatchEncoding

class PaperDataset(Dataset):
    def __init__(self, abstracts: BatchEncoding, titles: BatchEncoding):
        # Convert dictionary to T5 input format
        self.input_ids = abstracts.input_ids
        self.attention_mask = abstracts.attention_mask
        self.labels = titles.input_ids

    def __getitem__(self, index: int) -> Any:
        return self.input_ids[index], self.attention_mask[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.input_ids)
