from sklearn.model_selection import StratifiedKFold

from src.config import CFG
from src.data_setup import dataloader
from src.utils import preprocess, train_loop, test_loop


class Train:
    def __init__(self, data_path, model, optimizer, metric):
        self.data_path = data_path
        self.model = model
        self.optimizer = optimizer
        self.metric = metric

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
            train = self.data.loc[train_index]
            val = self.data.loc[val_index]

            # Dataloader
            train_dataloader = dataloader(train, loader_type="train")
            val_dataloader = dataloader(val, loader_type="val")

            # Train process
            for epoch in range(CFG.epochs):
                train_loop(
                    model=self.model,
                    train_dataloader=train_dataloader,
                    optimizer=self.optimizer,
                )

                test_loop(
                    model=self.model, test_dataloader=val_dataloader, metric=self.metric
                )

                score = self.metric.compute().item()
                self.oof.append(score)

                print("\n")
                print("Epoch:", epoch)
                print("Score:", score)
                self.metric.reset()
