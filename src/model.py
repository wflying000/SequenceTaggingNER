import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import AutoModel


class EncoderSoftmaxForNer(nn.Module):
    def __init__(self, config):
        super(EncoderSoftmaxForNer, self).__init__()
        self.encoder = AutoModel.from_pretrained(config.pretrained_model_path)
        self.linear = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout = nn.Dropout(config.dropout)
        self.config = config

    def forward(self, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = encoder_output.last_hidden_state
        hidden_state = self.dropout(hidden_state)
        logits = self.linear(hidden_state)

        outputs = {
            "logits": logits,
        }

        return outputs
    
class EncoderCRFForNer(nn.Module):
    def __init__(self, config):
        super(EncoderCRFForNer, self).__init__()
        self.encoder = AutoModel.from_pretrained(config.pretrained_model_path)
        self.linear = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.dropout = nn.Dropout(config.dropout)
        self.config = config
    
    def forward(self, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = encoder_output.last_hidden_state
        hidden_state = self.dropout(hidden_state)
        logits = self.linear(hidden_state)
        
        loss = None
        if "labels" in inputs:
            labels = inputs["labels"]
            tags = torch.where(labels >= 0, labels, 0)
            loss = -self.crf(emissions=logits, tags=tags, mask=attention_mask)

        predictions = self.crf.decode(emissions=logits, mask=attention_mask)

        outputs = {
            "logits": logits,
            "loss": loss,
            "predictions": predictions,
        }
        
        return outputs

