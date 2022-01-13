import numpy as np
from torch.utils.data import Dataset

class PaperDataset(Dataset):
    def __init__(self, subset, tokenizer):
        # Convert to numpy array for tokenizer
        data = np.array(subset)

        # Seperate titles and abstracts
        titles, abstracts = data.T

        # Tokenize
        tokenized_abstracts = tokenizer.batch_encode_plus(abstracts, padding=True, truncation=True, return_tensors="pt")
        tokenized_titles = tokenizer.batch_encode_plus(titles, padding=True, truncation=True, return_tensors="pt")

        # Prepare for T5 input
        self.input_ids = tokenized_abstracts.input_ids
        self.attention_mask = tokenized_abstracts.attention_mask
        self.labels = tokenized_titles.input_ids

    def __getitem__(self, index):
        return self.input_ids[index], self.attention_mask[index], self.labels[index]

    def __len__(self):
        return len(self.input_ids)


import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

class ArvixDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

    def setup(self, stage: list[str] = None):
        
        if stage == "fit" or stage is None:
            self.train_set = torch.load('data/processed/train_set.py')
            self.val_set = torch.load('data/processed/val_set.pt')

        if stage == "test":
            self.test_set = torch.load('data/processed/test_set.pt')

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.tesst_set, batch_size=32)