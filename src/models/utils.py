import os
from datetime import datetime as dt
import time

import torch
from google.cloud import storage


def setup_folders(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    # We set the environment variable such that we have google
    # authentication it could also be done with google oauth i think
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "Bucket upload: File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )


def save_model(model, BUCKET="title-generation-bucket"):
    # PARAMS
    RUN_TIME = dt.now().strftime("%m%d_%H%M_%S")

    # If we dont have a model folder we create one
    setup_folders("models")

    # Save checkpoint
    ckpt_path = f"models/T5_model_{RUN_TIME}.pth"
    checkpoint = model.state_dict()
    torch.save(checkpoint, ckpt_path)

    # We save the training curve
    print("Model checkpoint was saved locally at:", ckpt_path)

    # save the model checkpoint to cloud
    for _ in range(10):
        try:
            if BUCKET is not None:
                bucket_name = BUCKET
                source_file_name = ckpt_path
                destination_blob_name = f"title-generation/{ckpt_path}"
                upload_blob(bucket_name, source_file_name, destination_blob_name)
                print(
                    "Bucket upload: Model checkpoint was saved in GCP at:",
                    f"{bucket_name}/{ckpt_path}",
                )
            break
        except Exception:
            print('Failed bucket upload, trying again in 10s.')
            time.sleep(10)
