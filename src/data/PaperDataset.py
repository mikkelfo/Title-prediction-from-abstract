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
