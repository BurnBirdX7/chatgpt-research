import statistics
import random

import faiss  # type: ignore
from typing import Dict, Tuple
from transformers import RobertaTokenizer, RobertaModel  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
from progress.bar import ChargingBar  # type: ignore

from src import Roberta, Config, EmbeddingsBuilder, Index, Wiki


def estimate_thresholds(index: Index,
                        data: Dict[str, str],
                        tokenizer: RobertaTokenizer,
                        model: RobertaModel) -> Tuple[float, float]:
    count = len(data) // 10
    pages = random.choices(list(data.items()), k=count)
    embedding_builder = EmbeddingsBuilder(tokenizer, model, normalize=True)
    embedding_builder.suppress_progress = True

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

    if Config.show_plot:
        plt.hist(positives, bins=50, alpha=0.5, label='Positives')
        plt.hist(negatives, bins=50, alpha=0.5, label='Negatives')
        plt.axvline(pos_mean, color='blue', label=f'Positive Mean = {pos_mean:.4f}')
        plt.axvline(neg_mean, color='red', label=f'Negative Mean = {neg_mean:.4f}')
        plt.legend()
        plt.xlabel('Cos Distance')
        plt.ylabel('Count')
        plt.show()

    return pos_mean, neg_mean


def main():
    tm = Roberta.get_default()

    print('Reading index...', end='')
    index = Index.load(Config.index_file, Config.mapping_file, Config.threshold)
    print('Done')

    data = dict()
    for page in ChargingBar("Loading related articles").iter(Config.page_names):
        data |= Wiki.parse(page)

    for page in ChargingBar("Loading unrelated articles").iter(Config.unrelated_page_names):
        data |= Wiki.parse(page)

    p, n = estimate_thresholds(index, data, *tm)
    print(f'Positive threshold: {p}')
    print(f'Negative threshold: {n}')


if __name__ == '__main__':
    main()
