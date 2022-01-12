
import click
import logging
from pathlib import Path
#from dotenv import find_dotenv, load_dotenv
import torch
import pandas as pd
from omegaconf import OmegaConf
from torch.utils.data import random_split
from transformers import T5Tokenizer

from src.data.PaperDataset import PaperDataset

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def make_dataset(input_filepath='data/raw', output_filepath='data/processed', tokenizer=None):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    if tokenizer is None:
        tokenizer = T5Tokenizer.from_pretrained('t5-base')

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

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
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    make_dataset()
