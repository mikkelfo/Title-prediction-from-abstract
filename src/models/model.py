from typing import Tuple

import torch.nn as nn
from torch import BoolTensor, FloatTensor, LongTensor, Tensor
from transformers import T5ForConditionalGeneration, T5Tokenizer


class PredNet(nn.Module):
    """
    Custom class which implements t5 with an option to freeze parameters
    """

    def __init__(self) -> None:
        super(PredNet, self).__init__()

        # We download a pretrained bert model from
        # https://huggingface.co/t5-base
        self.t5 = T5ForConditionalGeneration.from_pretrained("t5-base")

        # We freeze the parameters in T5. Speeds up training and prevents
        # updating model weights during fine-tuning
        # for param in self.t5.parameters():
        #     param.requires_grad = False

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: FloatTensor = None,
        decoder_input_ids: LongTensor = None,
        decoder_attention_mask: BoolTensor = None,
        labels: LongTensor = None,
    ) -> Tuple[FloatTensor]:
        x = self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        # t5 automatically generates decoder_input_ids and
        # decoder_attention_mask from labels

        return x

    def generate(self, input_ids: Tensor):
        # TODO: add output type
        return self.t5.generate(input_ids)


if __name__ == "__main__":
    model = PredNet()

    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    string = (
        "This is a test string that should be summarized using "
        "abstractive summarization by the T5 model."
    )
    summ = "Test string summarized by T5 model abstractive summary"
    input_ids = tokenizer.encode(string, return_tensors="pt")
    labels = tokenizer.encode(summ, return_tensors="pt")

    x = model(input_ids=input_ids, labels=labels)
    print(x)
    print(x.loss)
    x.loss.backward()
