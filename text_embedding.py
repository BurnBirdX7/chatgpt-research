import math
from typing import List, Optional

import numpy as np
from transformers import RobertaTokenizer, RobertaModel
import torch


def input_ids_embedding(input_ids: List[int],
                        model: RobertaModel) -> np.ndarray:
    sequence_length: int = model.config.max_position_embeddings - 2

    embeddings: Optional[np.ndarray] = None
    for i in range(0, len(input_ids), sequence_length):
        input_ids_tensor = torch.tensor(input_ids[i:i + sequence_length]).unsqueeze(0).to(model.device)
        output = model(input_ids_tensor)
        seq_embeddings = output.last_hidden_state.detach().squeeze(0).cpu().numpy()
        if embeddings is None:
            embeddings = seq_embeddings
        else:
            embeddings = np.concatenate([embeddings, seq_embeddings])

    assert embeddings.shape[0] == len(input_ids)

    return embeddings


def text_embedding(text: str,
                   tokenizer: RobertaTokenizer,
                   model: RobertaModel) -> np.ndarray:

    input_ids = tokenizer.encode(text)
    return input_ids_embedding(input_ids, model)
