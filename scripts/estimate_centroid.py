import sys
from typing import List, Iterable

import numpy as np
import pandas as pd

from scripts._elvis_data import elvis_unrelated_articles, elvis_unrelated_articles_large
from src import EmbeddingsBuilder, Roberta
from src.online_wiki import OnlineWiki
from transformers import RobertaTokenizer, RobertaModel     # type: ignore
from progress.bar import ChargingBar                        # type: ignore

from src.config import EmbeddingBuilderConfig


def estimate_centroid(data: Iterable[str], tokenizer: RobertaTokenizer, model: RobertaModel) -> np.ndarray:
    embedding_config = EmbeddingBuilderConfig(tokenizer=tokenizer,
                                              model=model,
                                              normalize=False,
                                              centroid_file=None)

    embedding_builder = EmbeddingsBuilder(embedding_config)
    embedding_builder.suppress_progress_report = True

    embeddings = np.empty((0, embedding_builder.embedding_length))
    for page in ChargingBar('Processing texts').iter(data):
        embeddings = np.concatenate([embeddings, embedding_builder.from_text(page)])

    print(" >> Embedding count:", len(embeddings))

    return embeddings.mean(0)


def estimate_centroid_for_elvis_persona(tokenizer: RobertaTokenizer, model: RobertaModel) -> np.ndarray:
    page_names = elvis_unrelated_articles + elvis_unrelated_articles_large
    texts: List[str] = []
    for name in ChargingBar('Loading articles').iter(page_names):
        texts += OnlineWiki.get_sections(name).values()

    return estimate_centroid(texts, tokenizer, model)


def estimate_centroid_for_tsv_data(filepath: str, selection_size: int, tokenizer: RobertaTokenizer, model: RobertaModel) -> np.ndarray:
    df = pd.read_csv(filepath, delimiter='\t', names=['pid', 'text']).sample(selection_size)
    return estimate_centroid(df['text'], tokenizer, model)


def estimate_centroid_script(centroid_file, *args):
    if len(sys.argv) == 2:
        centroid_data = estimate_centroid_for_elvis_persona(*Roberta.get_default())
    elif len(sys.argv) == 4:
        centroid_data = estimate_centroid_for_tsv_data(args[0], int(args[1]), *Roberta.get_default())
    else:
        print("Usage:\n"
              "\tpython -m scripts estimate_centroid <output_file>\n"
              "or\n"
              "\tpython -m scripts estimate_centroid <output_file> <path_to_passage_collection> <selection_size>")
        return

    np.save(centroid_file, centroid_data)


if __name__ == '__main__':
    estimate_centroid_script(sys.argv[1], *sys.argv[2:])