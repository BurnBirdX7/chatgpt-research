from __future__ import annotations

import datetime
import time
from dataclasses import dataclass
from typing import Callable

import pandas as pd
import torch.cuda

from src.chaining import Chain
from scripts.coloring_pipeline import get_extended_coloring_pipeline


def majority_label(tokens: list[str], pos2chain: dict[int, Chain]) -> bool:
    return len(pos2chain) >= 0.5 * len(tokens)


def percent_label(percent: float) -> Callable:
    def __func(tokens: list[str], pos2chain: dict[int, Chain]) -> bool:
        return len(pos2chain) >= (percent * len(tokens) / 100)

    __func.__name__ = f"{percent}_percent_label"
    return __func


def longest_chain_percent_label(percent: float) -> Callable:
    def __func(tokens: list[str], pos2chain: dict[int, Chain]) -> bool:
        max_chain = max(pos2chain.values(), key=lambda ch: len(ch))
        return len(max_chain) >= (percent * len(tokens) / 100)

    __func.__name__ = f"longest_chain_{percent}_percent_label"
    return __func


incremental = (
    [majority_label]
    + [percent_label(x) for x in range(10, 91, 10)]
    + [longest_chain_percent_label(x) for x in range(10, 91, 10)]
)


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
pipeline.assert_prerequisites()


def estimate():
    start_time = time.time()

    # Sample passages
    all_passages = pd.read_csv("passages.csv")
    pos_passages = all_passages[all_passages["supported"]].sample(100)
    neg_passages = all_passages[~all_passages["supported"]].sample(100)
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
    return False


if __name__ == "__main__":
    run = True
    while run:
        try:
            run = estimate()
        except Exception as e:
            torch.cuda.empty_cache()
            print("Caught exception:", e)
