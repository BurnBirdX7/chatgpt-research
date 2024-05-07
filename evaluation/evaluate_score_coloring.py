from __future__ import annotations

import logging
import pathlib
from typing import List

import ujson
import typing as t
import numpy as np

import pandas as pd

from src.chaining.nodes import CollectTokenScoreNode
from src.pipeline import Pipeline, PipelineResult
from src.score_coloring_pipeline import ScoreColoringPipeline
from evaluation.roc_curve import roc_curve

logger = logging.getLogger(__name__)


def _score_cover(target: float):
    def score_cover(res: PipelineResult) -> float:
        probs = res.last_node_result
        return (probs >= target).sum().astype(float) / float(len(probs))

    target_s = str(target).replace(".", "_")
    score_cover.__name__ = f"score_cover_{target_s}"

    return score_cover


def _pipeline_setup() -> Pipeline:
    pipeline = ScoreColoringPipeline("max-score")
    pipeline.store_intermediate_data = False
    pipeline.force_caching("input-tokenized")

    def _max(slice_: List[float]) -> np.float32:
        return np.max(slice_).astype(np.float32)

    node = t.cast(CollectTokenScoreNode, pipeline.nodes["scores"])
    node.score_func = _max

    return pipeline


def start(output: pathlib.Path):
    passages = pd.read_csv("selected_passages.csv")
    with open("progress.json", "r") as f:
        start_idx = int(ujson.load(f)["idx"])

    logger.info(f"Starting evaluation with progress counter on {start_idx}")

    pipeline = _pipeline_setup()
    statistics = [_score_cover(0.01), _score_cover(0.001), _score_cover(0.0001)]
    roc_curve(pipeline, statistics, passages, start_idx, output)
