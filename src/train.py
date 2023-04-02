from transformers import AdamW
from torchmetrics.classification import MulticlassF1Score
from sklearn.model_selection import StratifiedKFold

from src.config import CFG
from src.data_setup import dataloader
from src.model import CustomModel
from src.utils import preprocess, train_loop, test_loop


class Train:
    def __init__(self, data_path):
        self.data_path = data_path

        self.skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=44)
        self.oof = []

    def _prepare(self):
        self.data = preprocess(data_path=self.data_path)

    def execute(self):
        # Prepare data
        self._prepare()

        # Cross validation
        for idx, (train_index, val_index) in enumerate(
            self.skf.split(self.data["text"], self.data["labels"])
        ):
            # Data split
            train = self.data.loc[train_index].reset_index().drop("index", axis=1)
            val = self.data.loc[val_index].reset_index().drop("index", axis=1)

            # Dataloader
            train_dataloader = dataloader(train, loader_type="train")
            val_dataloader = dataloader(val, loader_type="val")

            # Model 
            model = CustomModel().to(CFG.device)
            optimizer = AdamW(model.parameters(), lr=CFG.learning_rate)
            metric = MulticlassF1Score(num_classes=CFG.num_labels, average="macro").to(
                CFG.device
            )

            # Train process
            for epoch in range(CFG.epochs):
                train_loop(
                    model=model, train_dataloader=train_dataloader, optimizer=optimizer
                )

                test_loop(model=model, test_dataloader=val_dataloader, metric=metric)

                score = metric.compute().item()
                self.oof.append(score)

                print("\n")
                print("Epoch:", epoch)
                print("Score:", score)
                metric.reset()
