import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data_setup import CustomDataset
from src.model import CustomModel
from src.config import CFG


class CustomPipeline:
    def __init__(self, data):
        self.data = data
        self.model = CustomModel()
        self.model.load_state_dict(
            torch.load("src/model/model.pt", map_location=torch.device("cpu"))[
                "best_model"
            ],
            strict=False,
        )
        self.tokenizer = AutoTokenizer.from_pretrained("src/model/")

    def _preprocess(self):
        texts = self.data["text"].values.tolist()
        encodings = self.tokenizer(texts, truncation=True, padding=True)
        labels = torch.ones(len(texts)).type(torch.LongTensor)

        dataset = CustomDataset(encodings, labels)
        dataloader = DataLoader(dataset, batch_size=CFG.batch_size, shuffle=False)

        return dataloader

    def _forward(self, dataloader):
        predictions = []
        self.model.eval()
        with torch.inference_mode():
            for batch in dataloader:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )

                predictions.append(torch.argmax(outputs.logits, dim=1))
        return torch.cat(predictions).tolist()

    def postprocess(self):
        dataloader = self._preprocess()
        predictions = self._forward(dataloader)

        label_map = {
            "0": "",
            "1": "INSULT",
            "2": "PROFANITY",
            "3": "SEXIST",
            "4": "RACIST",
        }

        is_offensive = []
        target = []
        for label in predictions:
            result = label_map[str(label)]
            if result == "":
                is_offensive.append(0)
            else:
                is_offensive.append(1)
            target.append(result)

        offensive_col = self.data.filter(regex="off").columns.tolist()[0]
        self.data[offensive_col] = is_offensive
        self.data.target = target

        return self.data
