from model import PredNet
from transformers import T5Tokenizer
import torch
import pandas as pd


def train(epoch, model, optimizer, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()

    total_loss = 0
    
    for step, batch in enumerate(dataloader):
        # progress update after every 50 batches.
        if step % 1 == 0 and not step == 0:
            print("> Training Batch {:>5,}  of  {:>5,}.".format(step, len(dataloader)))

        # Clear previously gradients
        optimizer.zero_grad()

        batch = [r.to(device) for r in batch]
        input_ids, labels = batch

        loss = model(input_ids=input_ids, labels=labels).loss
        
        # T5 uses CrossEntropyLoss internally
        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch: {epoch:3}     Loss: {total_loss:3.2f}')

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, tokenizer):
        self.df = pd.read_csv(csv_path)[:100]
        self.titles = tokenizer.batch_encode_plus(self.df['titles'].tolist(), return_tensors="pt", padding=True, truncation=True).input_ids
        self.abstracts = tokenizer.batch_encode_plus(self.df['summaries'].tolist(), return_tensors="pt", padding=True, truncation=True).input_ids

    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        title, abstract = self.titles[index], self.abstracts[index]
        return title, abstract



if __name__ == '__main__':
    model = PredNet()
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)

    dataset = CustomDataset('data/raw/arxiv_data.csv', tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5)


    train(0, model, optimizer, dataloader)

