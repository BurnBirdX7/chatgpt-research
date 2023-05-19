import statistics
import random

from typing import Dict, Tuple
from transformers import RobertaTokenizer, RobertaModel  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
import faiss  # type: ignore
from progress.bar import Bar  # type: ignore

from IntervalToSource import IntervalToSource
from test.text_embedding import text_embedding
from wiki import parse_wiki
import roberta
import config


def estimate_thresholds(index: faiss.Index,
                        mapping: IntervalToSource,
                        data: Dict[str, str],
                        tokenizer: RobertaTokenizer,
                        model: RobertaModel) -> Tuple[float, float]:
    count = len(data) // 10
    pages = random.choices(list(data.items()), k=count)

    positives = []
    negatives = []

    for src, text in pages:
        embeddings = text_embedding(text, tokenizer, model)
        faiss.normalize_L2(embeddings)
        dists, ids = index.search(embeddings, 1)
        for dist, id in zip(dists, ids):
            found_src = mapping.get_source(id[0])
            if src == found_src:
                positives.append(dist[0])
            else:
                negatives.append(dist[0])

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
        plt.xlim(0.85, 1.0)
        plt.show()

    return pos_mean, neg_mean


def main():
    print('Reading index...', end='')
    index = faiss.read_index(config.index_file)
    if config.faiss_use_gpu:
        index = faiss.index_cpu_to_gpu(index)
    print('Done')

    data = dict()
    for page in Bar("Loading related articles").iter(config.page_names):
        data |= parse_wiki(page)

    for page in Bar("Loading unrelated articles").iter(config.unrelated_page_names):
        data |= parse_wiki(page)

    mapping = IntervalToSource.read_csv(config.ranges_file)
    tm = roberta.get_default()

    p, n = estimate_thresholds(index, mapping, data, *tm)
    print(f'Positive threshold: {p}')
    print(f'Negative threshold: {n}')


if __name__ == '__main__':
    main()