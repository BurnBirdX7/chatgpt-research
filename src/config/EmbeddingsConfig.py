from dataclasses import dataclass
from typing import Optional

from transformers import RobertaTokenizer, RobertaModel  # type: ignore

from .BaseConfig import BaseConfig, DefaultValue
from ..Roberta import Roberta


@dataclass
class EmbeddingsConfig(BaseConfig):
    tokenizer: RobertaTokenizer = DefaultValue(Roberta.get_default()[0])
    model: RobertaModel = DefaultValue(Roberta.get_default()[1])
    normalize: bool = DefaultValue(False)                           # type: ignore
    centroid_file: Optional[str] = DefaultValue("centroid.npy")     # type: ignore

