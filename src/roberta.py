from typing import Tuple, Dict

import transformers
from transformers import RobertaTokenizer, RobertaModel, RobertaForMaskedLM  # type: ignore
import torch  # type: ignore


class Roberta:
    _static_model: Dict[str, RobertaModel] = {}
    _static_mlm_model: Dict[str, RobertaForMaskedLM] = {}
    _static_tokenizer: Dict[str, RobertaTokenizer] = {}

    @classmethod
    def _get_device(cls) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def get_default(cls, model_name: str = "roberta-large") -> Tuple[RobertaTokenizer, RobertaModel]:
        return cls.get_default_tokenizer(model_name), cls.get_default_model(model_name)

    @classmethod
    def get_default_tokenizer(cls, model_name: str = "roberta-large") -> transformers.RobertaTokenizer:
        if model_name not in cls._static_tokenizer:
            t = RobertaTokenizer.from_pretrained(model_name)
            cls._static_tokenizer[model_name] = t
            return t

        return cls._static_tokenizer[model_name]

    @classmethod
    def get_default_model(cls, model_name: str = "roberta-large") -> transformers.RobertaModel:
        if model_name not in cls._static_model:
            m = RobertaModel.from_pretrained(model_name).to(cls._get_device())
            cls._static_model[model_name] = m
            return m
        return cls._static_model[model_name]

    @classmethod
    def get_default_masked_model(cls, model_name: str = "roberta-large") -> transformers.RobertaForMaskedLM:
        if model_name not in cls._static_mlm_model:
            m = RobertaForMaskedLM.from_pretrained(model_name).to(cls._get_device())
            cls._static_mlm_model[model_name] = m
            return m
        return cls._static_mlm_model[model_name]
