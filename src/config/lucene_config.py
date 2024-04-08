import os
from dataclasses import dataclass, field

from .base_config import BaseConfig


@dataclass
class LuceneConfig(BaseConfig):
    index_path: str = field(default_factory=lambda: os.environ.get("LUCENE_INDEX_PATH"))
