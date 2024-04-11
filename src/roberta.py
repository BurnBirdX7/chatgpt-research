from typing import Tuple, Dict
from transformers import RobertaTokenizer, RobertaModel  # type: ignore
import torch  # type: ignore


class Roberta:
    static_storage: Dict[str, Tuple[RobertaTokenizer, RobertaModel]] = dict()

    @classmethod
    def get_default(
        cls, model_name: str = "roberta-large"
    ) -> Tuple[RobertaTokenizer, RobertaModel]:
        if model_name not in cls.static_storage:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            t = RobertaTokenizer.from_pretrained(model_name)
            m = RobertaModel.from_pretrained(model_name).to(device)

            cls.static_storage[model_name] = (t, m)
            return t, m

        return cls.static_storage[model_name]

    @classmethod
    def get_default_tokenizer(cls, model_name: str = "roberta-large"):
        return cls.get_default(model_name)[0]

    @classmethod
    def get_default_model(cls, model_name: str = "roberta-large"):
        return cls.get_default(model_name)[1]
