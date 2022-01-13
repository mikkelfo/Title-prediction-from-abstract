import click
import torch
import pandas as pd
from omegaconf import OmegaConf
from torch.utils.data import random_split
from transformers import T5Tokenizer

from src.data.PaperDataset import PaperDataset

@click.command()
@click.argument('input_filepath', type=click.Path(), default='data/raw')
@click.argument('output_filepath', type=click.Path(), default='data/processed')
def make_dataset(input_filepath, output_filepath, tokenizer=None):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    if tokenizer is None:
        tokenizer = T5Tokenizer.from_pretrained('t5-base')

    # get configuration
    config = OmegaConf.load('src/data/config.yaml')

    # Read csv file into pandas (only specified columns)
    data = pd.read_csv(f'{input_filepath}/{config.data_name}', usecols=[config.title_column, config.abstract_column])
    data = data.to_numpy()

    # Split the data
    train, val, test = random_split(data, [config.n_train, config.n_val, config.n_test], generator=torch.Generator().manual_seed(1337))
    train_set = PaperDataset(train, tokenizer)
    val_set = PaperDataset(val, tokenizer)
    test_set = PaperDataset(test, tokenizer)

    # store data
    torch.save(train_set, f'{output_filepath}/train_set.pt')
    torch.save(val_set, f'{output_filepath}/val_set.pt')
    torch.save(test_set, f'{output_filepath}/test_set.pt')

if __name__ == '__main__':
    make_dataset()
