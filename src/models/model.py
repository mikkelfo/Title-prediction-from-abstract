from typing import Any, List, Tuple

import torch
from pytorch_lightning import LightningModule
from torch import FloatTensor, LongTensor
from transformers import T5ForConditionalGeneration


class PredNet(LightningModule):
    """
    Custom class which implements t5 with an option to freeze parameters
    """

    def __init__(self) -> None:
        super(PredNet, self).__init__()

        # We download a pretrained bert model from
        # https://huggingface.co/t5-base
        self.t5 = T5ForConditionalGeneration.from_pretrained("t5-base")

        # We freeze the all layers in T5 (but the last called 'lm_head').
        # Speeds up training and prevents updating model weights during fine-tuning
        for param in list(self.t5.parameters())[:-1]:
            param.requires_grad = False

    def forward(
        self,
        input_ids: LongTensor,
        attention_mask: FloatTensor,
        labels: LongTensor,
    ) -> Tuple[FloatTensor]:
        x = self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        # t5 automatically generates decoder_input_ids and
        # decoder_attention_mask from labels
        return x

    def training_step(self, batch: Any, batch_idx: int) -> FloatTensor:
        input_ids, attention_mask, labels = batch
        loss = self(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        ).loss
        self.log("training_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> FloatTensor:
        input_ids, attention_mask, labels = batch
        loss = self(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        ).loss
        self.log("training_loss", loss, on_epoch=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> FloatTensor:
        input_ids, attention_mask, labels = batch
        loss = self(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        ).loss
        self.log("training_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=0.001, weight_decay=0.001)

    def generate(self, input_ids) -> List[int]:
        return self.t5.generate(input_ids)


if __name__ == "__main__":
    model = PredNet()
