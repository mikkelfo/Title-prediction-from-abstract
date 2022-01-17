from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

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
    trainer = Trainer(max_epochs=1, logger=wandb_logger)
    trainer.fit(model, dm)
    save_model(model)
