from typing import TypeVar

import click
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import random_split
from torch.utils.data.dataset import Subset
from transformers import T5Tokenizer

from src.data.PaperDataset import PaperDataset

T = TypeVar('T')  # from source code to torch.utils.data.dataset

@click.command()
@click.argument("input_filepath", type=click.Path(), default="data/raw")
@click.argument("output_filepath", type=click.Path(), default="data/processed")
def make_dataset(
    input_filepath: str, output_filepath: str
) -> None:
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    # get configuration
    config = OmegaConf.load("src/data/config.yaml")

    # Read csv file into pandas (only specified columns)
    data = pd.read_csv(
        f"{input_filepath}/{config.data_name}",
        usecols=[config.title_column, config.abstract_column]
    )
    data = data.to_numpy()

    # Split the data
    train, val, test = random_split(
        data,
        [config.n_train, config.n_val, config.n_test],
        generator=torch.Generator().manual_seed(1337),
    )

    # store data
    store_data(train, f"{output_filepath}/train", tokenizer)
    store_data(val, f"{output_filepath}/val", tokenizer)
    store_data(test, f"{output_filepath}/test", tokenizer)

def store_data(
    data_set: Subset[T], output_filepath: str, tokenizer: T5Tokenizer = None
) -> None:

    # Seperate titles and abstracts
    data_set = np.array(data_set)
    titles, abstracts = data_set.T

    # Tokenize
    tokenized_abstracts = tokenizer.batch_encode_plus(
            abstracts, padding=True, truncation=True, return_tensors="pt"
    )
    tokenized_titles = tokenizer.batch_encode_plus(
        titles, padding=True, truncation=True, return_tensors="pt"
    )
    
    # Prepare for T5 input
    input_ids = tokenized_abstracts.input_ids
    attention_mask = tokenized_abstracts.attention_mask
    labels = tokenized_titles.input_ids

    for i in range(len(data_set)):
        torch.save(
            {'input_id': input_ids[i].clone(), 
             'attention_mask': attention_mask[i].clone(), 
             'label': labels[i].clone()}, 
            f"{output_filepath}/{i}.pt"
        )

if __name__ == "__main__":
    make_dataset()
