import click
import torch
import pandas as pd
from omegaconf import OmegaConf
from torch.utils.data import random_split
from transformers import T5Tokenizer

@click.command()
@click.argument('input_filepath', type=click.Path(), default='data/raw')
@click.argument('output_filepath', type=click.Path(), default='data/processed')
def make_dataset(input_filepath: str, output_filepath: str) -> None:
    """ 
        Turns raw csv file into a dictionary of tokenized data
    """
    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    # get configuration
    config = OmegaConf.load('src/data/config.yaml')

    # Read csv file into pandas (only specified columns)
    data = pd.read_csv(f'{input_filepath}/{config.data_name}', usecols=[config.title_column, config.abstract_column])
    data = data.to_numpy()

    # Add tokenizing
    titles, abstracts = data.T
    tokenized_abstracts = tokenizer.batch_encode_plus(abstracts, padding=True, truncation=True, return_tensors="pt")
    tokenized_titles = tokenizer.batch_encode_plus(titles, padding=True, truncation=True, return_tensors="pt")

    # Create dictionary and save to file
    torch.save({'titles': tokenized_titles, 'abstracts': tokenized_abstracts}, output_filepath + '/tokenized_data.pt')

if __name__ == '__main__':
    make_dataset()
