from typing import Optional

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, random_split
from transformers import T5Tokenizer

from src.data.PaperDataset import PaperDataset


class ArvixDataModule(pl.LightningDataModule):
    def __init__(self, config: str = "src/data/config.yaml") -> None:
        super().__init__()
        self.config = OmegaConf.load(config)

    def prepare_data(self) -> None:
        # Add tokenizing
        tokenizer = T5Tokenizer.from_pretrained("t5-base")

        titles, abstracts = torch.load("data/processed/data.pt").T
        tokenized_abstracts = tokenizer.batch_encode_plus(
            abstracts, padding=True, truncation=True, return_tensors="pt"
        )
        tokenized_titles = tokenizer.batch_encode_plus(
            titles, padding=True, truncation=True, return_tensors="pt"
        )

        self.data = PaperDataset(tokenized_abstracts, tokenized_titles)

    def setup(self, stage: Optional[str] = None):
        train, val, test = random_split(
            self.data,
            [self.config.n_train, self.config.n_val, self.config.n_test],
            generator=torch.Generator().manual_seed(1337),
        )

        if stage == "fit" or stage is None:
            self.train_set = train
            self.val_set = val

        if stage == "test":
            self.test_set = test

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=32, num_workers=4)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size=32, num_workers=4)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, batch_size=32, num_workers=4)


if __name__ == "__main__":
    dm = ArvixDataModule()
