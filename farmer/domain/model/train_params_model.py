from dataclasses import dataclass, field
from typing import Dict


@dataclass
class TrainParams:
    # 最適化のために探索するパラメータ
    model_name: str = None
    backbone: str = None
    activation: str = "softmax"
    freeze: bool = False
    loss: Dict[str, dict] = field(default_factory=dict)
    classification_class_weight: Dict[str, float] = field(default_factory=dict)
    batch_size: int = 1
    weights_info: Dict[str, str] = field(default_factory=dict)
    optimizer: str = None
    learning_rate: float = None
    aux_learning_rate: float = None
    augmentation: Dict[str, int] = field(default_factory=dict)
    augmix: bool = False
    opt_decay: float = 0.001
    scheduler: dict = None
