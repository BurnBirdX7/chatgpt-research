from dataclasses import dataclass, field
from typing import List

from .base_config import BaseConfig


@dataclass
class WikiConfig(BaseConfig):
    target_pages: List[str] = field(default_factory=list)
