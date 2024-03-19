from dataclasses import dataclass
from typing import List

from .base_config import BaseConfig


@dataclass
class WikiConfig(BaseConfig):
    target_pages: List[str]
