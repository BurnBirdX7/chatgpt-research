from dataclasses import dataclass

from .base_config import BaseConfig, DefaultValue
from src.config import IndexConfig
from typing import Dict


@dataclass
class ThresholdConfig(IndexConfig):
    show_plot: bool = DefaultValue(True)                    # type: ignore
    model_name: str = DefaultValue("roberta-large")         # type: ignore
    data: Dict[str, str] = DefaultValue(dict())   # type: ignore
