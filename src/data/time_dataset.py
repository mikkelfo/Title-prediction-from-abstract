import argparse
import time
import numpy as np
from torch.utils.data import DataLoader
from PaperDataset import PaperDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_workers', default=0, type=int)
    parser.add_argument('-batch_size', default=512, type=int)
    args = parser.parse_args()

    dataset = PaperDataset('train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=args.num_workers)
    
    res = [ ]
    for _ in range(5):
        start = time.time()
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx > 100:
                break
        end = time.time()

        res.append(end - start)
        
    res = np.array(res)
    print(f'Timing: {np.mean(res)}+-{np.std(res)}')