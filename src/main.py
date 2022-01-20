from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping

from src.data.dataModule import ArvixDataModule
from src.models.model import PredNet
from src.models.utils import save_model
import os

if __name__ == "__main__":
    # Team API key
    os.environ["WANDB_API_KEY"] = "XX"

    wandb_logger = WandbLogger(project = "arxiv", entity = "title-generation")
    dm = ArvixDataModule()
    model = PredNet()
    early_stopping_callback = EarlyStopping(
        monitor="validation_loss", patience=3, verbose=True, mode="min"
    )
    trainer = Trainer(max_epochs=10, logger=wandb_logger, accelerator="gpu", gpus=1,
                      callbacks=[early_stopping_callback])
    trainer.fit(model, dm)
    save_model(model)
