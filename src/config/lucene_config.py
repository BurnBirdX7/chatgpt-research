import os
from dataclasses import dataclass, field

from .base_config import BaseConfig


@dataclass
class LuceneConfig(BaseConfig):

    @staticmethod
    def get_from_env() -> str:
        if "LUCENE_INDEX_PATH" not in os.environ:
            raise ValueError("LUCENE_INDEX_PATH environment variable is not set")

        return os.environ["LUCENE_INDEX_PATH"]

    index_path: str = field(default_factory=LuceneConfig.get_from_env)  # type: ignore
