import time

import numpy as np
import torch
from model import PredNet
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from train_model import train
from transformers import T5Tokenizer

from src.data.PaperDataset import PaperDataset

if __name__ == '__main__':
    if torch.cuda.is_available():
        model = PredNet().cuda()
    else:
        model = PredNet()
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.001)

    dataset = PaperDataset('train')
    dataloader = DataLoader(dataset, batch_size=512)
    res = [ ]
    for _ in range(5):
        start = time.time()
        train(0)
        end = time.time()

        res.append(end - start)
        
    res = np.array(res)
    print(f'Timing: {np.mean(res)}+-{np.std(res)}')
    train(0)