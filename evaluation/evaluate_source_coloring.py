from __future__ import annotations

import functools
import pathlib

import ujson
import logging
import typing as t

import pandas as pd

from src.pipeline import PipelineResult
from src.source_coloring_pipeline import SourceColoringPipeline

from evaluation.binary_stats import estimate_bool
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


def prob2bool(func: t.Callable[[PipelineResult], float]) -> t.List[t.Callable[[PipelineResult], bool]]:
    wrappers = []
    for threshold in range(5, 96, 10):

        @functools.wraps(func)
        def wrapper(res: PipelineResult):
            return func(res) >= threshold

        wrapper.__name__ = func.__name__ + f"_tr{threshold}"
        wrappers.append(wrapper)
    return wrappers


statistics = [overall_cover, longest_chain_cover]
incremental = prob2bool(overall_cover) + prob2bool(longest_chain_cover)


def _prepare_pipeline() -> SourceColoringPipeline:
    pipeline = SourceColoringPipeline.new_extended()
    pipeline.store_intermediate_data = False
    pipeline.force_caching("input-tokenized")
    return pipeline


def start_roc(output: pathlib.Path):
    passages = pd.read_csv("selected_passages.csv")
    with open("progress.json", "r") as f:
        start_idx = int(ujson.load(f)["idx"])

    logger.info(f"Starting evaluation with progress counter on {start_idx}")

    roc_curve(_prepare_pipeline(), statistics, passages, start_idx, output)


def start_bool():
    passages = pd.read_csv("selected_passages.csv")

    with open("progress.json", "r") as f:
        start_idx = int(ujson.load(f)["idx"])

    estimate_bool(_prepare_pipeline(), incremental, passages, start_idx)