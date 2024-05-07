import collections
import dataclasses
import logging
import pathlib
import typing as t

import pandas as pd
import ujson

from src.pipeline import Pipeline, PipelineResult


logger = logging.getLogger(__name__)
StatTuple = collections.namedtuple("StatTuple", ["func", "value"])
_filename = ".eval_progress.binary.json"


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
        return Stat(**d)

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


def estimate_bool(
    pipeline: Pipeline,
    statistics: t.Sequence[t.Callable[[PipelineResult], bool]],
    passages: pd.DataFrame,
    start_idx: int,
    output: pathlib.Path,
):
    with open(_filename, "r") as file:
        progress = ujson.load(file)

    stats = {f.__name__: Stat() for f in statistics} | {k: Stat.from_dict(v) for k, v in progress["stats"].items()}

    print(f"{stats.keys()=}")

    for i, passage, supported in passages[start_idx:].itertuples():
        logger.info(f"Evaluating {i + 1} / {len(passages)}...")
        result = pipeline.run(passage)
        for func in statistics:
            stats[func.__name__].add(func(result), supported)

        with open(_filename, "w") as file:
            progress["idx"] = i + 1
            progress["stats"] = {k: dataclasses.asdict(v) for k, v in stats.items()}
            ujson.dump(progress, file, indent=2)

    bests = {
        "precision": StatTuple("", 0.0),
        "recall": StatTuple("", 0.0),
        "accuracy": StatTuple("", 0.0),
        "f1": StatTuple("", 0.0),
    }

    key: str
    stat: Stat
    for key, stat in stats.items():
        for val_name, tup in bests.items():
            val: float = getattr(stat, val_name)
            if val > tup[1]:
                bests[val_name] = StatTuple(key, val)

    data = {
        f"best_{val_name}": {
            "name": name,
            "value": val,
            "stats": dataclasses.asdict(stats[name]),
            "accuracy": stats[name].accuracy,
            "recall": stats[name].recall,
            "precision": stats[name].precision,
            "f1": stats[name].f1,
        }
        for val_name, (name, val) in bests.items()
        if name in stats
    }

    path = output.joinpath("binary_stats.json")
    with open(path, "w") as f:
        ujson.dump(data, f)
    logger.info(f"Data saved to {path}")
