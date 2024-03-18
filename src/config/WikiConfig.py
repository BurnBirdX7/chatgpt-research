from dataclasses import dataclass
from typing import List

from .BaseConfig import BaseConfig


@dataclass
class WikiConfig(BaseConfig):
    target_pages: List[str]
