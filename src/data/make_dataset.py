# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
#from dotenv import find_dotenv, load_dotenv
import torch
import pandas as pd
import numpy as np
from omegaconf import OmegaConf

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    np.random.seed(1337)
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # get configuration
    config = OmegaConf.load('src/data/config.yaml')
    data_name = config.data_name
    n_data    = config.n_data
    n_train   = int(n_data * config.r_train)
    n_val     = int(n_data * config.r_val)

    # get indices for data split
    # into train set, validation set, and test set
    index = np.arange(config.n_data)
    np.random.shuffle(index)
    train_index = index[:n_train]
    val_index   = index[n_train:n_train + n_val]
    test_index  = index[n_train + n_val:]

    # get data and split
    data = pd.read_csv(f'{input_filepath}/{data_name}')
    data = data.to_numpy()
    train_data = data[train_index]
    val_data   = data[val_index]
    test_data  = data[test_index]

    # store data
    torch.save(train_data, f'{output_filepath}/train_data.pt')
    torch.save(val_data, f'{output_filepath}/val_data.pt')
    torch.save(test_data, f'{output_filepath}/test_data.pt')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()
