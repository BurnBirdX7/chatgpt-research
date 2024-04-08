from dataclasses import dataclass, field

from src.config import IndexConfig
from typing import Dict


@dataclass
class ThresholdConfig(IndexConfig):
    show_plot: bool = field(default=True)
    model_name: str = field(default="roberta-large")
    data: Dict[str, str] = field(default_factory=dict)
