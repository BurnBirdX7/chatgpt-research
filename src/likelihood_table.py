import dataclasses


import torch
import typing as t

import numpy as np
import numpy.typing as npt

from src import Roberta
from src.embeddings_builder import TokenizeTextNode
from src.config.embeddings_config import EmbeddingBuilderConfig


@dataclasses.dataclass
class Result:
    source_ids: t.List[int]
    source_tokens: t.List[str]
    target_ids: t.List[int]
    target_tokens: t.List[str]

    table: npt.NDArray[np.float32]


def likelihood_table(source_text: str, target_text: str) -> Result:
    model = Roberta.get_default_masked_model()
    tokenizer = Roberta.get_default_tokenizer()

    tokenizer_output = tokenizer(
        text=source_text,
        return_tensors="pt",
        return_attention_mask=True,
        padding=True,
        pad_to_multiple_of=tokenizer.model_max_length,
    )

    def make_batch(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape((-1, tokenizer.model_max_length)).to(model.device)

    source_attention_mask = make_batch(tokenizer_output["attention_mask"])
    source_token_id_batch = make_batch(tokenizer_output["input_ids"])
    with torch.no_grad():
        # Logits have dimensions: (passage, position, vocab)
        batched_logits = model(source_token_id_batch, attention_mask=source_attention_mask).logits

    batched_likelihoods = torch.nn.functional.softmax(batched_logits, dim=2)
    # Remove padding tokens and cut special tokens
    likelihoods = batched_likelihoods[source_attention_mask > 0.0][1:-1].cpu().numpy()
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]

    node = TokenizeTextNode("tokenize", EmbeddingBuilderConfig(tokenizer=tokenizer))

    return Result(
        source_ids=source_token_id_batch[source_attention_mask > 0.0][1:-1].tolist(),
        source_tokens=node.process(source_text),
        target_ids=target_ids,
        target_tokens=node.process(target_text),
        table=likelihoods[:, target_ids],
    )
