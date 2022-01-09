import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch



class PredNet(nn.Module):
    def __init__(self) -> None:
        super(PredNet, self).__init__()

        # We download a pretrained bert model from https://huggingface.co/t5-base
        self.t5 = T5ForConditionalGeneration.from_pretrained('t5-base')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')

        # We freeze the parameters in T5. Speeds up training and prevents updating model weights during fine-tuning
        for param in self.t5.parameters():
            param.requires_grad = False

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 768)
        self.fc2 = nn.Linear(768, 768)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None):
        # First pass it through T5
        x = self.t5(        
            input_ids=input_ids,
            labels=labels
        )
        return x
        # Then through the dense layers (fine-tuning)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

if __name__ == '__main__':
    model = PredNet()
    input_ids = model.tokenizer.encode('This is test string. Try to summarize this bad boy', return_tensors="pt")
    labels = model.tokenizer.encode('Test string', return_tensors="pt")
    x = model(input_ids=input_ids, labels=labels)
    # print(x)
    zzz = model.t5.generate(input_ids, max_length=10, min_length=2)
    print(zzz)
    y = model.tokenizer.decode(zzz[0])

    print()
    
    
