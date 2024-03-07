import os
from dataclasses import dataclass

from .BaseConfig import BaseConfig, DefaultValue


@dataclass
class LuceneConfig(BaseConfig):
    index_path: str = DefaultValue(os.environ.get("LUCENE_INDEX_PATH"))  # type: ignore
