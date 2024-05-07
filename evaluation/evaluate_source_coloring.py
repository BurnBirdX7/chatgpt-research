from __future__ import annotations

import functools
import pathlib

import ujson
import logging
import typing as t

import pandas as pd

from src.pipeline import PipelineResult, Pipeline
from src.pipeline.pipeline_group import PipelineGroup
from src.source_coloring_pipeline import SourceColoringPipeline

from evaluation.binary_stats import estimate_bool
from evaluation.roc_curve import roc_curve
from evaluation.stats import collect_statistics

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
    wrappers: t.List[t.Callable[[PipelineResult], bool]] = []
    for threshold in range(5, 96, 10):

        @functools.wraps(func)
        def wrapper(res: PipelineResult) -> bool:
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
    with open(".eval_progress.roc.json", "r") as f:
        start_idx = int(ujson.load(f)["idx"])

    logger.info(f"Starting evaluation with progress counter on {start_idx}")

    roc_curve(_prepare_pipeline(), statistics, passages, start_idx, output)


def start_bool(output: pathlib.Path):
    passages = pd.read_csv("selected_passages.csv")

    with open(".eval_progress.binary.json", "r") as f:
        start_idx = int(ujson.load(f)["idx"])

    estimate_bool(_prepare_pipeline(), incremental, passages, start_idx, output)


def start_stats(output: pathlib.Path):
    passages = pd.read_csv("selected_passages.csv")

    with open(".eval_progress.stats.json", "r") as f:
        start_idx = int(ujson.load(f)["idx"])

    def _pipeline_preset(name: str, use_bidirectional_chaining: bool) -> Pipeline:
        pipeline = SourceColoringPipeline.new_extended(name, use_bidirectional_chaining)
        pipeline.force_caching("input-tokenized")
        pipeline.store_intermediate_data = False
        return pipeline

    # GLOBALS
    unidir_pipeline = _pipeline_preset("unidirectional", use_bidirectional_chaining=False)
    bidir_pipeline = _pipeline_preset("bidirectional", use_bidirectional_chaining=True)
    pipeline_group = PipelineGroup("all-chains", [unidir_pipeline, bidir_pipeline])

    collect_statistics(pipeline_group, passages, start_idx, output)
