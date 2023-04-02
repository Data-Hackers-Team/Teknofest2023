import pandas as pd
from torch import tensor
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from src.config import CFG


class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: tensor(value[idx]) for key, value in self.encodings.items()}
        item["labels"] = tensor(self.labels[idx])
        return item


def dataloader(data: pd.DataFrame, loader_type: str = "train"):
    assert loader_type in ["train", "val", "test"]

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_path)

    # Create encodings
    encodings = tokenizer(data["text"].values.tolist(), truncation=True, padding=True)

    # Dataset setup
    dataset = CustomDataset(encodings, data["labels"])

    # Dataloader setup
    return DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=(True if loader_type == "train" else False),
    )
