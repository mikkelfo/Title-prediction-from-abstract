import torch
from model import PredNet
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from transformers import T5Tokenizer
import torch
from src.data.PaperDataset import PaperDataset
from datetime import datetime as dt
from google.cloud import storage
import os

# We set the environment variable such that we have google authentication it could also be done with google oauth i think
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"

def upload_blob(bucket_name, source_file_name, destination_blob_name):
        """Uploads a file to the bucket."""
        # The ID of your GCS bucket
        # bucket_name = "your-bucket-name"
        # The path to your file to upload
        # source_file_name = "local/path/to/file"
        # The ID of your GCS object
        # destination_blob_name = "storage-object-name"

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(source_file_name)

        print(
            "Bucket upload: File {} uploaded to {}.".format(
                source_file_name, destination_blob_name
            )
        )

def train(
    epoch: int, model: PredNet, optimizer: Optimizer, dataloader: DataLoader
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    total_loss = 0

    for step, batch in enumerate(dataloader):
        # progress update after every 50 batches.
        if step % 5 == 0 and not step == 0:
            print(f"> Training Batch {step:>5,}  of  {len(dataloader):>5,}.")
            break

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

    # PARAMS
    RUN_TIME = dt.now().strftime("%m%d_%H%M_%S")
    BUCKET = "title-generation-bucket"

    # Save checkpoint
    ckpt_path = f'models/T5_model_{RUN_TIME}.pth'
    checkpoint = model.state_dict()
    torch.save(checkpoint, ckpt_path)

    # We save the training curve
    print("Model checkpoint was saved locally at:", ckpt_path)

    # save the model checkpoint to cloud
    if BUCKET is not None:
        bucket_name = BUCKET
        source_file_name = ckpt_path
        destination_blob_name = f'title-generation/{ckpt_path}'
        upload_blob(bucket_name, source_file_name, destination_blob_name)
        print("Bucket upload: Model checkpoint was saved in GCP at:", f'{bucket_name}/{ckpt_path}')

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
