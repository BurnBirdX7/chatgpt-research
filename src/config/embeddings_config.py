from dataclasses import dataclass, field
from typing import Optional

from transformers import RobertaTokenizer, RobertaModel  # type: ignore

from .base_config import BaseConfig
from ..roberta import Roberta


@dataclass
class EmbeddingBuilderConfig(BaseConfig):
    tokenizer: RobertaTokenizer = field(default_factory=Roberta.get_default_tokenizer)
    model: RobertaModel = field(default_factory=Roberta.get_default_model)
    normalize: bool = field(default=False)
    centroid_file: Optional[str] = field(default="centroid.npy")
