import os
from dataclasses import dataclass

from BaseConfig import BaseConfig, DefaultValue, Default  # type: ignore


@dataclass
class LuceneConfig(BaseConfig):
    index_path: Default[str] = DefaultValue(os.environ.get("LUCENE_INDEX_PATH"))  # type: ignore
