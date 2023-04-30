from typing import Tuple

from transformers import RobertaTokenizer, RobertaModel
import torch
import config

def get_default() -> Tuple[RobertaTokenizer, RobertaModel]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    t = RobertaTokenizer.from_pretrained(config.model_name)
    m = RobertaModel.from_pretrained(config.model_name).to(device)
    return t, m
