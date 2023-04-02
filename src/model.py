from torch.nn import Module
from transformers import AutoModelForSequenceClassification

from src.config import CFG


class CustomModel(Module):
    def __init__(self):
        super().__init__()

        self.model_path = CFG.model_path
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.model_path, num_labels=CFG.num_labels
        )

    def forward(self, input_ids, attention_mask, labels):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
