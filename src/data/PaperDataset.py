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

def Arxiv(data_filepath, batch_size = 64):
        train_dataset = PaperDataset(data_filepath, "train")
        test_dataset = PaperDataset(data_filepath, "test")
        val_dataset = PaperDataset(data_filepath, "val")

        train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)

        test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)

        val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)

        return(train_loader, test_loader, val_loader)

# Just temporary while it is local
data_filepath = "data/processed"
batch_size = 1

trainloader, testloader, valloader = Arxiv(data_filepath, batch_size)

abstracts, titles = next(iter(trainloader))

print(abstracts)
print(titles)

# We want to move this to the cloud

# Not sure if this really worked out