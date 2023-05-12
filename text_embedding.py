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
    sequence_length: int = model.config.max_position_embeddings
    shorten_by: int = 2 if sequence_length % 2 == 0 else 3
    sequence_length -= shorten_by
    window_step: int = sequence_length // 2

    embedding_len: int = model.config.hidden_size
    embeddings: np.ndarray = np.empty((0, embedding_len))
    previous_half: np.ndarray | None = None
    for i in Bar('Computing').iter(range(0, len(input_ids), window_step)):
        # Create tensor with acceptable dimensions:
        input_ids_tensor = torch.tensor(input_ids[i : i + sequence_length]).unsqueeze(0)

        # Moves tensor to model's device
        input_ids_tensor = input_ids_tensor.to(model.device)

        output = model(input_ids_tensor)
        seq_embeddings = output.last_hidden_state.detach().squeeze(0).cpu().numpy()

        if previous_half is not None:
            # Get mean value of 2 halves (prev[:t] and curr[t:])
            current_half = (previous_half + seq_embeddings[:window_step]) / 2
            embeddings = np.concatenate([embeddings, current_half], dtype=np.float32)
        else:
            embeddings = seq_embeddings[window_step:].astype(np.float32)

        previous_half = seq_embeddings[window_step:]

    if previous_half is not None:
        embeddings = np.concatenate([embeddings, previous_half], dtype=np.float32)

    count, length = embeddings.shape
    assert count == len(input_ids)
    assert length == embedding_len
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
