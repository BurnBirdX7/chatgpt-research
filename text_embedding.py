import numpy as np
from progress.bar import Bar  # type: ignore
import torch  # type: ignore

from transformers import RobertaTokenizer, RobertaModel  # type: ignore
from typing import List, Optional

"""
Functions:
Compute embeddings for the given input
"""


def input_ids_embedding(input_ids: List[int], model: RobertaModel) -> np.ndarray:
    """
    :param input_ids: list of token ids
    :param model: model to build embeddings with
    :return: numpy array with dimensions (token_count, embedding_length)
    """
    sequence_length: int = model.config.max_position_embeddings - 2

    embeddings: np.ndarray = np.empty((0, model.config.hidden_size))
    for i in Bar('Computing').iter(range(0, len(input_ids), sequence_length)):
        # Create tensor with acceptable dimensions:
        input_ids_tensor = torch.tensor(input_ids[i : i + sequence_length]).unsqueeze(0)

        # Moves tensor to model's device
        input_ids_tensor = input_ids_tensor.to(model.device)

        output = model(input_ids_tensor)
        seq_embeddings = output.last_hidden_state.detach().squeeze(0).cpu().numpy()
        embeddings = np.concatenate([embeddings, seq_embeddings], dtype=np.float32)
    assert embeddings.shape[0] == len(input_ids)
    return embeddings


def text_embedding(
    text: str, tokenizer: RobertaTokenizer, model: RobertaModel
) -> np.ndarray:
    """
    :param text: input
    :param tokenizer: tokenizer to split text in tokens
    :param model: model to build embeddings with
    :return: numpy array with dimensions (token_count, embedding_length)
             token_count depends on text contents
    """

    input_ids = tokenizer.encode(text)
    return input_ids_embedding(input_ids, model)
