import torch
import pandas as pd
from tqdm import tqdm

from src.config import CFG


def preprocess(data_path: str) -> pd.DataFrame:
    # Read data
    data = pd.read_csv(data_path, sep="|")

    # Drop rows that has 1 char in text column
    data = data.drop(data.loc[data.text.apply(lambda x: len(x) == 1)].index)

    # Fix 'is_offensive' value for mismatch rows
    data.loc[(data.is_offensive == 1) & (data.target == "OTHER"), "is_offensive"] = 0

    # Label encoding
    data.loc[data.target == "OTHER", "target"] = 0
    data.loc[data.target == "INSULT", "target"] = 1
    data.loc[data.target == "PROFANITY", "target"] = 2
    data.loc[data.target == "SEXIST", "target"] = 3
    data.loc[data.target == "RACIST", "target"] = 4

    # Label data type conversion
    data.target = data.target.astype(int)

    # Prepare for model
    data.drop(["id", "is_offensive"], axis=1, inplace=True)
    data.rename(columns={"target": "labels"}, inplace=True)
    data = data.reset_index().drop("index", axis=1)

    return data


def train_loop(model, train_dataloader, optimizer):
    # Prepare model for training
    model.train()

    for batch in tqdm(train_dataloader):
        # Prepare model inputs
        input_ids = batch["input_ids"].to(CFG.device)
        attention_mask = batch["attention_mask"].to(CFG.device)
        labels = batch["labels"].to(CFG.device)

        # Get model outputs
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        # Set gradients to zero
        optimizer.zero_grad()

        # Calculate loss
        loss = outputs.loss
        loss.backward()

        # Perform optimization
        optimizer.step()


def test_loop(model, test_dataloader, metric):
    # Prepare model for testing (or validation)
    model.eval()

    with torch.inference_mode():
        for batch in tqdm(test_dataloader):
            # Prepare model inputs
            input_ids = batch["input_ids"].to(CFG.device)
            attention_mask = batch["attention_mask"].to(CFG.device)
            labels = batch["labels"].to(CFG.device)

            # Get model outputs
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            # Calculate score
            logits = torch.argmax(outputs.logits, dim=1)
            f1_score = metric(logits, labels).item()
