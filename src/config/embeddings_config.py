from dataclasses import dataclass
from typing import Optional

from transformers import RobertaTokenizer, RobertaModel  # type: ignore

from .base_config import BaseConfig, DefaultValue
from ..roberta import Roberta


@dataclass
class EmbeddingsConfig(BaseConfig):
    tokenizer: RobertaTokenizer = DefaultValue(None)
    model: RobertaModel = DefaultValue(None)
    normalize: bool = DefaultValue(False)                           # type: ignore
    centroid_file: Optional[str] = DefaultValue("centroid.npy")     # type: ignore

    def __post_init__(self):
        BaseConfig.__post_init__(self)
        if self.tokenizer is None:
            self.tokenizer = Roberta.get_default()[0]
        if self.model is None:
            self.model = Roberta.get_default()[1]
