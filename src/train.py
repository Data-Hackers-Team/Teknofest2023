import os
from torch.optim.swa_utils import AveragedModel, SWALR
from transformers import AdamW, get_linear_schedule_with_warmup
from torchmetrics.classification import MulticlassF1Score
from sklearn.model_selection import StratifiedKFold

from src.config import CFG
from src.data_setup import dataloader
from src.model import CustomModel
from src.utils import preprocess, train_val_fn, set_random_seed


class Train:
    def __init__(self, data_path):
        self.data_path = data_path

        self.skf = StratifiedKFold(n_splits=5)
        self.oof = []

    def _prepare(self):
        self.data = preprocess(data_path=self.data_path)

    def execute(self):
        # Prepare data
        self._prepare()

        set_random_seed(42)

        # Cross validation
        for fold, (train_index, test_index) in enumerate(
            self.skf.split(self.data["text"], self.data["labels"])
        ):
            train = self.data.loc[train_index].reset_index().drop("index", axis=1)
            test = self.data.loc[test_index].reset_index().drop("index", axis=1)

            train_dataloader = dataloader(train, loader_type="train")
            test_dataloader = dataloader(test, loader_type="test")

            model = CustomModel().to(CFG.device)

            optimizer = AdamW(model.parameters(), lr=5e-5)
            metric = MulticlassF1Score(num_classes=CFG.num_labels, average="macro").to(
                CFG.device
            )

            swa_model = AveragedModel(model).to(CFG.device)
            swa_scheduler = SWALR(optimizer, swa_lr=5e-5)

            train_steps = int(len(train) / CFG.batch_size * 3)
            scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer, num_warmup_steps=50, num_training_steps=train_steps
            )

            os.mkdir(f"src/pretrained_model/model_{fold}/")
            os.mkdir(f"src/pretrained_model/model_swa_{fold}/")

            for epoch in range(3):
                print("\n")
                print("Epoch", epoch)

                train_val_fn(
                    model=model,
                    optimizer=optimizer,
                    train_dataloader=train_dataloader,
                    val_dataloader=test_dataloader,
                    fold=fold,
                    epoch=epoch,
                    early_stop_threshold=50,
                    swa_model=swa_model,
                    swa_scheduler=swa_scheduler,
                    scheduler=scheduler,
                    swa_step=True,
                )

                print("\n")
                print("F1 Score:", metric.compute().item())

                metric.reset()
