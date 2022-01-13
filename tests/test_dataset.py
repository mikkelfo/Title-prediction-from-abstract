import os
from omegaconf import OmegaConf
import torch
import random

''' 
    Requirements:
        make_dataset.py must be ran before these tests
'''

# Initial test of a succesful run of make_dataset.py
assert os.path.isfile('data/processed/test_set.pt'), 'Test set does not exist'
assert os.path.isfile('data/processed/train_set.pt'), 'Train set file does not exist'
assert os.path.isfile('data/processed/val_set.pt'), 'Validation set file does not exist'

# Load in relevant objects
config = OmegaConf.load('src/data/config.yaml')
test_set = torch.load('data/processed/test_set.pt')
train_set = torch.load('data/processed/train_set.pt')
val_set = torch.load('data/processed/val_set.pt')

# Test sizes of datasets
assert len(test_set) == config.n_test
assert len(train_set) == config.n_train
assert len(val_set) == config.n_val
assert len(test_set) + len(train_set) + len(val_set) == config.n_data

# Check datapoint size
for dataset in [test_set, train_set, val_set]:
    assert len(random.choice(dataset)) == 3, 'Dataset __getitem__ returns incorrect number of things'
    input_id, attention_mask, label = random.choice(test_set)
    assert attention_mask.max() == 1, 'Dataset returns incorrect attention_mask'    # Very important to ensure the ordering of (input_id, attention_mask and label) correct
    assert len(input_id) == len(attention_mask), 'Input id does not match the size of attention mask'





