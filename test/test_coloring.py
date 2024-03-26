import pprint
from dataclasses import dataclass

import pandas as pd

from src.token_chain import Chain
from scripts.color_pipeline import get_coloring_pipeline

def majority_label(tokens: list[str], pos2chain: dict[int, Chain]) -> bool:
    return len(pos2chain) >= 0.5 * len(tokens)

def percent_label(percent: float) -> callable:
    def __func(tokens: list[str], pos2chain: dict[int, Chain]) -> bool:
        return len(pos2chain) >= (percent * len(tokens) / 100)

    __func.__name__ = f"{percent}_percent_label"
    return __func

def longest_chain_percent_label(percent: float) -> callable:
    def __func(tokens: list[str], pos2chain: dict[int, Chain]) -> bool:
        max_chain = max(pos2chain.values(), key=lambda ch: len(ch))
        return len(max_chain) >= (percent * len(tokens) / 100)

    __func.__name__ = f"longest_chain_{percent}_percent_label"
    return __func


incremental = ([majority_label] +
               [percent_label(x) for x in range(10, 91, 10)] +
               [longest_chain_percent_label(x) for x in range(10, 91, 10)])

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
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else None

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn > 0 else None


pipeline = get_coloring_pipeline()
pipeline.force_caching("input-tokenized")
pipeline.check_prerequisites()

if __name__ == '__main__':
    passages = pd.read_csv('passages.csv')

    stats = {
        f.__name__: Stat()
        for f in incremental
    }

    print(f"{stats.keys()=}")

    for i, passage, supported in passages.sample(100).itertuples():
        pos2chain, _, cache = pipeline.run(passage)
        tokens = cache['input-tokenized']

        for f in incremental:
            pred = f(tokens, pos2chain)
            stats[f.__name__].add(pred, supported)

    for key, stat in stats.items():
        print(f"{key}:")
        print(f"\t{stat}")
        print(f"\t{stat.precision=}")
        print(f"\t{stat.recall=}")
        print(f"\t{stat.accuracy=}")
