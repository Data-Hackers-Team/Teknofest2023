from dataclasses import dataclass


@dataclass
class CFG:
    model_path: str = "dbmdz/bert-base-turkish-cased"
    max_length: int = 300
    epochs: int = 3
    batch_size: int = 8
    device: str = "cuda"
    num_labels: int = 5
    learning_rate: float = 5e-5
