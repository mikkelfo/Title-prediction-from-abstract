import torch
from model import PredNet
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from transformers import T5Tokenizer
import torch
from src.data.PaperDataset import PaperDataset


from src.data.PaperDataset import PaperDataset


def train(
    epoch: int, model: PredNet, optimizer: Optimizer, dataloader: DataLoader
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    total_loss = 0

    for step, batch in enumerate(dataloader):
        # progress update after every 50 batches.
        if step % 1 == 0 and not step == 0:
            print(f"> Training Batch {step:>5,}  of  {len(dataloader):>5,}.")

        # Clear previously gradients
        optimizer.zero_grad()

        batch = [r.to(device) for r in batch]
        input_ids, attention_mask, labels = batch

        loss = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        ).loss

        # T5 uses CrossEntropyLoss internally
        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch: {epoch:3}     Loss: {total_loss:3.2f}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        model = PredNet().cuda()
    else:
        model = PredNet()
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.001)

    dataset = PaperDataset('train')
    dataloader = DataLoader(dataset, batch_size=5)

    train(0, model, optimizer, dataloader)
