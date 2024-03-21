import statistics
import random

import faiss  # type: ignore
from typing import Dict, Tuple
from transformers import RobertaTokenizer, RobertaModel  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
from progress.bar import ChargingBar  # type: ignore

from scripts._elvis_data import elvis_related_articles, elvis_unrelated_articles
from src import Roberta, EmbeddingsBuilder, Index, OnlineWiki
from src.config import EmbeddingBuilderConfig, ThresholdConfig


def estimate_thresholds_on_index(index: Index,
                                 tokenizer: RobertaTokenizer,
                                 model: RobertaModel,
                                 config: ThresholdConfig) -> Tuple[float, float]:
    """
    Estimates thresholds based on provided data

    :param index: That stores data
    :param tokenizer: RoBERTa tokenizer instance
    :param model: RoBERTa model instance
    """

    count = len(config.data) // 10
    pages = random.choices(list(config.data.items()), k=count)
    embedding_builder = EmbeddingsBuilder(EmbeddingBuilderConfig(tokenizer, model, normalize=True))
    embedding_builder.suppress_progress_report = True

    positives = []
    negatives = []

    for real_src, text in ChargingBar("Processing pages").iter(pages):
        embeddings = embedding_builder.from_text(text)
        sources, dists = index.get_embeddings_source(embeddings)
        for dist, found_src in zip(dists, sources):
            if real_src == found_src:
                positives.append(dist)
            else:
                negatives.append(dist)

    pos_mean = statistics.mean(positives)
    neg_mean = statistics.mean(negatives)

    if config.show_plot:
        plt.hist(positives, bins=50, alpha=0.5, label='Positives')
        plt.hist(negatives, bins=50, alpha=0.5, label='Negatives')
        plt.axvline(pos_mean, color='blue', label=f'Positive Mean = {pos_mean:.4f}')
        plt.axvline(neg_mean, color='red', label=f'Negative Mean = {neg_mean:.4f}')
        plt.legend()
        plt.xlabel('Cos Distance')
        plt.ylabel('Count')
        plt.show()

    return pos_mean, neg_mean


def estimate_thresholds(config: ThresholdConfig):
    tm = Roberta.get_default()

    print('Reading index...', end='')
    index = Index.load(config)
    print('Done')

    p, n = estimate_thresholds_on_index(index, tm[0], tm[1], config)
    print(f'Positive threshold: {p}')
    print(f'Negative threshold: {n}')


def estimate_thresholds_for_elvis():
    data = dict[str, str]()
    for page in ChargingBar("Loading related articles").iter(elvis_related_articles):
        data |= OnlineWiki.get_sections(page)

    for page in ChargingBar("Loading unrelated articles").iter(elvis_unrelated_articles):
        data |= OnlineWiki.get_sections(page)

    estimate_thresholds(ThresholdConfig(data=data))


if __name__ == '__main__':
    estimate_thresholds_for_elvis()
