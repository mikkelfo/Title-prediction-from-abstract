from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from src.data.dataModule import ArvixDataModule
from src.models.model import PredNet
from src.models.utils import save_model



if __name__ == '__main__':
    wandb_logger = WandbLogger()
    dm = ArvixDataModule()
    model = PredNet()
    trainer = Trainer(max_epochs=2, logger=wandb_logger)
    trainer.fit(model, dm)

    save_model(model)