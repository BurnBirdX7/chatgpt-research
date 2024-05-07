from __future__ import annotations

import datetime
import functools
import pathlib

import ujson
import time
import logging
from dataclasses import dataclass
from typing import Callable, List

import pandas as pd

from src.pipeline import Pipeline, PipelineResult
from src.chaining import ElasticChain
from src.source_coloring_pipeline import SourceColoringPipeline

from evaluation.roc_curve import roc_curve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def overall_cover(res: PipelineResult) -> float:
    tokens = res.cache["input-tokenized"]
    pos2chain = res.last_node_result
    return len(pos2chain) / len(tokens)


def longest_chain_cover(res: PipelineResult) -> float:
    tokens = res.cache["input-tokenized"]
    chains = res.last_node_result
    max_chain = max(chains, key=lambda ch: ch.target_len())
    return max_chain.target_len() / len(tokens)


def prob2bool(func: Callable) -> List[Callable]:
    wrappers = []
    for threshold in range(10, 91, 5):

        @functools.wraps(func)
        def wrapper(tokens: list[str], pos2chain: dict[int, ElasticChain]):
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
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else -1

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn > 0 else -1

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return 2 * p * r / (p + r) if p != -1 and r != -1 else -1


def estimate_bool(pipeline: Pipeline):
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

    key: str
    stat: Stat
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


def start(output: pathlib.Path):
    passages = pd.read_csv("selected_passages.csv")
    with open("progress.json", "r") as f:
        start_idx = int(ujson.load(f)["idx"])

    logger.info(f"Starting evaluation with progress counter on {start_idx}")

    pipeline = SourceColoringPipeline.new_extended()
    pipeline.store_intermediate_data = False
    pipeline.force_caching("input-tokenized")

    roc_curve(pipeline, statistics, passages, start_idx, output)
    exit(0)
