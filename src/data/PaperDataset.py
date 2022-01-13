import numpy as np
from torch import load
from torch import Tensor, FloatTensor, LongTensor
from torch.utils.data import Dataset
from typing import Tuple
from omegaconf import OmegaConf

class PaperDataset(Dataset):
    def __init__(self, subset: str) -> None:
        # Convert to numpy array for tokenizer
        self.subset = subset

        # get configuration
        config = OmegaConf.load("src/data/config.yaml")

        self.n = config[f"n_{subset}"]

    def __getitem__(
        self, index: int
    ) -> Tuple[Tensor, FloatTensor, LongTensor]:
        data_line = load(f"data/processed/{self.subset}-{index}.pt")
        return data_line['input_id'], data_line['attention_mask'], data_line['label']

    def __len__(self) -> int:
        return self.n
