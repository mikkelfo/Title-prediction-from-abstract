import torch
from torch.utils.data import Dataset, DataLoader
from omegaconf import OmegaConf

class PaperDataset(Dataset):
    def __init__(self, data_filepath, which = 'train'):
        # get configuration
        config = OmegaConf.load('src/data/config.yaml')

        # get dataset
        data = torch.load(f'{data_filepath}/{which}_data.pt')
        self.n_samples = data.shape[0]

        # get titles and abstracts
        self.titles = data[:, config.title_column]
        self.abstracts = data[:, config.abstract_column]

    def __getitem__(self, index):
        return self.abstracts[index], self.titles[index]

    def __len__(self):
        return self.n_samples
