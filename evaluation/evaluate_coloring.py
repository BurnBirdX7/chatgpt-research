from __future__ import annotations

import datetime
import functools
import ujson
import time
import logging
from dataclasses import dataclass
from typing import Callable, List, TextIO

import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

from src import QueryColbertServerNode
from src.chaining import Chain
from scripts.coloring_pipeline import get_extended_coloring_pipeline, get_coloring_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def overall_cover(tokens: list[str], pos2chain: dict[int, Chain]) -> float:
    return len(pos2chain) / len(tokens)


def longest_chain_cover(tokens: list[str], pos2chain: dict[int, Chain]):
    chains = pos2chain.values()
    max_chain = max(chains, key=lambda ch: len(ch))
    return len(max_chain) / len(tokens)


def prob2bool(func: Callable) -> List[Callable]:
    wrappers = []
    for threshold in range(10, 91, 5):
        @functools.wraps(func)
        def wrapper(tokens: list[str], pos2chain: dict[int, Chain]):
            prob = func(tokens, pos2chain)
            return prob >= threshold

        wrapper.__name__ = func.__name__ + f"_tr{threshold}"
        wrappers.append(wrapper)
    return wrappers


statistics = [overall_cover, longest_chain_cover]
incremental = prob2bool(overall_cover) + prob2bool(longest_chain_cover)


@dataclass
class Stat:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, pred: bool, true: bool) -> None:
        if pred and true:
            self.tp += 1
        elif pred and not true:
            self.fp += 1
        elif not pred and true:
            self.fn += 1
        else:
            self.tn += 1

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    @property
    def precision(self) -> float | None:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else None

    @property
    def recall(self) -> float | None:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn > 0 else None

    @property
    def f1(self) -> float | None:
        p = self.precision
        r = self.recall
        return 2 * p * r / (p + r) if p is not None and r is not None else None


pipeline = get_extended_coloring_pipeline()
pipeline.store_intermediate_data = False
pipeline.force_caching("input-tokenized")
queryNode: QueryColbertServerNode = pipeline.nodes['all-sources-dict-raw']

def roc_curve(passages: pd.DataFrame, start_idx: int):
    with open('progress.json', 'r') as f:
        preds = {
                    f.__name__: []
                    for f in statistics
                } | ujson.load(f)['preds']

    for i, x_test, y_true in passages[start_idx:].itertuples():
        logger.info(f"Evaluating {i + 1} / {len(passages)}...")
        res = pipeline.run(x_test)
        tokens = res.cache['input-tokenized']
        for func in statistics:
            preds[func.__name__].append(func(tokens, res.last_node_result))

        with open('progress.json', 'w') as f:
            ujson.dump({
                'idx': i+1,
                'preds': preds,
            }, f)

    for func, y_pred in preds.items():
        fpr, tpr, thresholds = metrics.roc_curve(passages['supported'], y_pred)
        roc_auc = metrics.auc(fpr, tpr)
        plt.title(f'ROC curve for {func.__name__} func')
        plt.plot(fpr, tpr, 'b', label=f'AUC = {roc_auc:0.2f}')
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()


def estimate_bool():
    start_time = time.time()

    # Sample passages
    all_passages = pd.read_csv("passages.csv")
    pos_passages = all_passages[all_passages["supported"]].sample(10)
    neg_passages = all_passages[~all_passages["supported"]].sample(10)
    passages = pd.concat([pos_passages, neg_passages], ignore_index=True)

    stats = {f.__name__: Stat() for f in incremental}

    print(f"{stats.keys()=}")

    for i, passage, supported in passages.itertuples():
        result = pipeline.run(passage)
        tokens = result.cache["input-tokenized"]

        for f in incremental:
            pred = f(tokens, result.last_node_result)
            stats[f.__name__].add(pred, supported)

    best_precision = ("", 0.0)
    best_recall = ("", 0.0)
    best_accuracy = ("", 0.0)
    best_f1 = ("", 0.0)

    for key, stat in stats.items():
        print(f"{key}:")
        print(f"\t{stat}")
        print(f"\t{stat.precision=}")
        print(f"\t{stat.recall=}")
        print(f"\t{stat.accuracy=}")
        print(f"\t{stat.f1=}")

        if stat.accuracy > best_accuracy[1]:
            best_accuracy = key, stat.accuracy

        if stat.recall > best_recall[1]:
            best_recall = key, stat.recall

        if stat.precision > best_precision[1]:
            best_precision = key, stat.precision

        if stat.f1 > best_f1[1]:
            best_f1 = key, stat.f1

    print(f"{best_accuracy=}")
    print(f"{best_precision=}")
    print(f"{best_recall=}")
    print(f"{best_f1=}")
    print(f"time elapsed: {datetime.timedelta(seconds=time.time() - start_time)}")


def start():
    passages = pd.read_csv("selected_passages.csv")
    with open("progress.json", "r") as f:
        start_idx = int(ujson.load(f)['idx'])

    logger.info(f"Starting evaluation with progress counter on {start_idx}")
    roc_curve(passages, start_idx)
    exit(0)
