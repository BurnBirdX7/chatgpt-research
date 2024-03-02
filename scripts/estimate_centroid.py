from typing import List, Iterable

import numpy as np

from scripts._elvis_data import elvis_unrelated_articles, elvis_unrelated_articles_large
from src import EmbeddingsBuilder, Wiki, Roberta
from transformers import RobertaTokenizer, RobertaModel     # type: ignore
from progress.bar import ChargingBar                        # type: ignore

from src.config.EmbeddingsConfig import EmbeddingsConfig
from src.config.WikiConfig import WikiConfig


def estimate_centroid(data: Iterable[str], tokenizer: RobertaTokenizer, model: RobertaModel) -> np.ndarray:
    embedding_config = EmbeddingsConfig(tokenizer=tokenizer,
                                        model=model,
                                        normalize=False,
                                        centroid_file=None)

    embedding_builder = EmbeddingsBuilder(embedding_config)
    embedding_builder.suppress_progress = True

    embeddings = np.empty((0, embedding_builder.embedding_length))
    for page in ChargingBar('Processing texts').iter(data):
        embeddings = np.concatenate([embeddings, embedding_builder.from_text(page)])

    return embeddings.mean(0)


def estimate_centroid_for_elvis_persona(centroid_file: str, tokenizer: RobertaTokenizer, model: RobertaModel) -> None:
    page_names = elvis_unrelated_articles + elvis_unrelated_articles_large
    texts: List[str] = []
    for name in ChargingBar('Loading articles').iter(page_names):
        texts += Wiki.parse(name).values()

    centroid_data = estimate_centroid(texts, tokenizer, model)
    np.save(centroid_file, centroid_data)

    print('Centroid:')
    print(repr(centroid_data))


if __name__ == '__main__':
    estimate_centroid_for_elvis_persona(*Roberta.get_default())
