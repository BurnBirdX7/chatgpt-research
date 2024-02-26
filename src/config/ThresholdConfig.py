from dataclasses import dataclass

from BaseConfig import BaseConfig, DefaultValue
from src.config.IndexConfig import IndexConfig


@dataclass
class ThresholdConfig(BaseConfig, IndexConfig):
    show_plot: bool = DefaultValue(True)             # type: ignore
    model_name: str = DefaultValue("roberta-large")  # type: ignore
    data: dict[str, str] = DefaultValue(dict())
