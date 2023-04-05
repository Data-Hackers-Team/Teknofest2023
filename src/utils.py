import os
import copy
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import CFG


def set_random_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def train_val_fn(
    model,
    optimizer,
    train_dataloader,
    val_dataloader,
    fold,
    epoch,
    early_stop_threshold,
    swa_model,
    swa_scheduler,
    scheduler,
    swa_step=False,
):
    model.train()

    best_score = 0
    for idx, batch in enumerate(train_dataloader):
        input_ids = batch["input_ids"].to(CFG.device)
        attention_mask = batch["attention_mask"].to(CFG.device)
        labels = batch["labels"].to(CFG.device)

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        optimizer.zero_grad()

        loss = outputs.loss
        loss.backward()

        optimizer.step()

        if swa_step:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        if (idx % 1000 == 0) or ((idx + 1) == len(train_dataloader)):

            score = val_fn(model, val_dataloader)
            print(score)
            print(best_score)

            if score > best_score:
                print(
                    f"FOLD: {fold}, Epoch: {epoch}, Batch {idx}, F1 = {round(score,4)}, checkpoint saved."
                )
                best_score = score
                early_stopping_counter = 0

                with torch.inference_mode():
                    best_model = copy.deepcopy(model.state_dict())
                    best_swa_model = copy.deepcopy(swa_model.state_dict())

                checkpoint = {
                    "model": model.state_dict(),
                    "best_model": best_model,
                    "best_metric": best_score,
                }
                torch.save(checkpoint, f"src/pretrained_model/model_{fold}/model.pt")

                if swa_step:
                    checkpoint = {
                        "model": model.state_dict(),
                        "best_model": best_swa_model,
                        "best_metric": best_score,
                    }
                    torch.save(
                        checkpoint, f"src/pretrained_model/model_swa_{fold}/model.pt"
                    )
            else:
                print(
                    f"FOLD: {fold}, Epoch: {epoch}, Batch {idx}, F1 = {round(score,4)}"
                )
                early_stopping_counter += 1
            if early_stopping_counter > early_stop_threshold:
                print(f"Early stopping triggered!")
                break


def val_fn(model, dataloader, metric):
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(dataloader):

            input_ids = batch["input_ids"].to(CFG.device)
            attention_mask = batch["attention_mask"].to(CFG.device)
            labels = batch["labels"].to(CFG.device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            logits = torch.argmax(outputs.logits, dim=1)
            f1_score = metric(logits, labels).item()

    return metric.compute().item()
