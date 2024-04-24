from dataclasses import dataclass, field
from typing import Optional

import transformers

from .base_config import BaseConfig
from ..roberta import Roberta


@dataclass
class EmbeddingBuilderConfig(BaseConfig):
    tokenizer: transformers.PreTrainedTokenizer = field(default_factory=Roberta.get_default_tokenizer)
    model: transformers.PreTrainedModel = field(default_factory=Roberta.get_default_model)
    normalize: bool = field(default=False)
    centroid_file: Optional[str] = field(default="centroid.npy")
