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
        labels, attention_mask, input_ids = batch

        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
        
        # T5 uses CrossEntropyLoss internally
        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch: {epoch:3}     Loss: {total_loss:3.2f}')


if __name__ == '__main__':
    model = PredNet().cuda()
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)

    dataset = torch.load('data/processed/train_set.pt')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5)


    train(0, model, optimizer, dataloader)

