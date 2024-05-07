import collections
import dataclasses
import datetime
import time
import typing as t

import pandas as pd
import ujson

from src.pipeline import Pipeline, PipelineResult


@dataclasses.dataclass
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

    @classmethod
    def from_dict(cls, d: t.Dict[str, int]) -> "Stat":
        return Stat(
            tp=d["tp"],
            tn=d["tn"],
            fp=d["fp"],
            fn=d["fn"],
        )

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


StatTuple = collections.namedtuple("StatTuple", ["func", "value"])

_filename = ".progress.binary.json"


def estimate_bool(
    pipeline: Pipeline,
    statistics: t.Sequence[t.Callable[[PipelineResult], bool]],
    passages: pd.DataFrame,
    start_idx: int,
):
    start_time = time.time()

    with open(_filename, "r") as f:
        progress = ujson.load(f)

    stats = {f.__name__: Stat() for f in statistics} | {k: Stat.from_dict(v) for k, v in progress["stats"].items()}

    print(f"{stats.keys()=}")

    for i, passage, supported in passages[start_idx:].itertuples():
        result = pipeline.run(passage)

        for f in statistics:
            pred = f(result)
            stats[f.__name__].add(pred, supported)

        with open(_filename, "w") as f:
            progress["idx"] = i + 1
            progress["stats"] = {k: dataclasses.asdict(v) for k, v in stats.items()}
            ujson.dump(progress, f, indent=2)

    best_precision = StatTuple("", 0.0)
    best_recall = StatTuple("", 0.0)
    best_accuracy = StatTuple("", 0.0)
    best_f1 = StatTuple("", 0.0)

    key: str
    stat: Stat
    for key, stat in stats.items():
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
