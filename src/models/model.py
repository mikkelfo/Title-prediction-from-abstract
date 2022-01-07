import torch.nn as nn
from transformers import AutoModel, pipeline



class PredNet(nn.Module):
    def __init__(self) -> None:
        super(PredNet, self).__init__()

        # We download a pretrained bert model from https://huggingface.co/t5-base
        self.t5 = AutoModel.from_pretrained('t5-base')
        self.tokenizer = AutoModel.from_pretrained('t5-base')

        # We freeze the parameters in T5. Speeds up training and prevents updating model weights during fine-tuning
        for param in self.t5.parameters():
            param.requires_grad = False

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 768)
        self.fc2 = nn.Linear(768, 768)

    def forward(self, x):
        # First pass it through T5
        x = self.t5(x)
        # Then through the dense layers (fine-tuning)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
