from transformers import AdamW
from torchmetrics.classification import MulticlassF1Score
from transformers import AutoModelForSequenceClassification

from src.config import CFG
from src.train import Train


if __name__ == "__main__":
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model_name_or_path=CFG.model_path, num_labels=CFG.num_labels
    ).to(CFG.device)

    obj = Train(
        data_path="teknofest_train_final.csv",
        model=model,
        optimizer=AdamW(model.parameters(), lr=CFG.learning_rate),
        metric=MulticlassF1Score(num_classes=CFG.num_labels, average="macro").to(
            CFG.device
        ),
    )
    obj.execute()
