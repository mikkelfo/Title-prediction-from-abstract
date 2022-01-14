from omegaconf import OmegaConf
import pandas as pd
import os


''' Test the config file '''
config = OmegaConf.load('src/data/config.yaml')
assert os.path.isfile(f'data/raw/{config.data_name}'), 'The data file does not exist'

# Load in the file
data = pd.read_csv(f'data/raw/{config.data_name}')
assert len(data) == config.n_data, 'Dataset did not have correct number of samples'
assert config.n_data == config.n_train + config.n_val + config.n_test, 'The sum of subset sizes does not match data size'
assert config.title_column in data.columns, '"titles" is not a header in the dataset'
assert config.abstract_column in data.columns, '"abstracts" is not a header in the dataset'
