import dataclasses
import logging
import pathlib
import typing as t

import numpy as np
import numpy.typing as npt
import pandas as pd

import ujson

from src import ElasticChain
from src.chaining import HardChain
from src.pipeline import PipelineResult
from src.pipeline.pipeline_group import PipelineGroup

_filename = ".eval_progress.stats.json"

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _Statistics:
    likelihood_gmeans: t.List[float]
    likelihood_ameans: t.List[float]
    scores: t.List[float]
    chain_count: int
    unique_source_count: int
    token_per_chain: t.List[float]
    text_token_len: int
    supports: bool

    @classmethod
    def from_dict(cls, d: dict) -> "_Statistics":
        return _Statistics(**d)


def _gmean(arr: npt.NDArray[np.float32]) -> float:
    return np.exp(np.log(arr).sum() / len(arr)).item()


def _collect(res: PipelineResult, _: bool) -> _Statistics:
    chains: t.List[ElasticChain | HardChain] = res.last_node_result
    tokens = res.cache["input-tokenized"]

    return _Statistics(
        likelihood_ameans=[np.mean(chain.likelihoods).item() for chain in chains],
        likelihood_gmeans=[_gmean(chain.likelihoods) for chain in chains],
        scores=[chain.get_score() for chain in chains],
        chain_count=len(chains),
        unique_source_count=len(set([chain.source for chain in chains])),
        token_per_chain=[chain.target_len() for chain in chains],
        text_token_len=len(tokens),
        supports=False,
    )


def collect_statistics(group: PipelineGroup, passages: pd.DataFrame, start_idx: int, output: pathlib.Path):
    with open(_filename, "r") as file:
        progress = ujson.load(file)

    collected_stats: t.Dict[str, t.List[_Statistics]] = {pipeline.name: [] for pipeline in group.pipelines} | {
        name: [_Statistics.from_dict(e) for e in l] for name, l in progress["p_stats"].items()
    }

    print(f"{collected_stats.keys()=}")

    for i, passage, supported in passages[start_idx:].itertuples():
        logger.info(f"Evaluating {i + 1} / {len(passages)}...")
        results: t.Dict[str, _Statistics]
        results, _ = group.run(passage, _collect)
        for pipeline_name, stats in results.items():
            stats.supports = supported
            collected_stats[pipeline_name].append(stats)

        with open(_filename, "w") as file:
            progress["idx"] = i + 1
            progress["p_stats"] = {k: [dataclasses.asdict(v) for v in lst] for k, lst in collected_stats.items()}
            ujson.dump(progress, file, indent=2)

    path = output.joinpath("stats.json")
    with open(path, "w") as f:
        ujson.dump({k: [dataclasses.asdict(v) for v in lst] for k, lst in collected_stats.items()}, f, indent=2)
    logger.info(f"Data written to {path}")
