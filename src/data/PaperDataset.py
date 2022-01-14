from typing import Tuple

from omegaconf import OmegaConf
from torch import FloatTensor, LongTensor, Tensor, load
from torch.utils.data import Dataset


class PaperDataset(Dataset):
    def __init__(self, subset: str) -> None:
        # Convert to numpy array for tokenizer
        self.subset = subset

        # get configuration
        config = dict(OmegaConf.load("src/data/config.yaml"))

        self.n = config[f"n_{subset}"]

    def __getitem__(self, index: int) -> Tuple[Tensor, FloatTensor, LongTensor]:
        data_line = load(f"data/processed/{self.subset}/{index}.pt")
        return data_line["input_id"], data_line["attention_mask"], data_line["label"]

    def __len__(self) -> int:
        return self.n
