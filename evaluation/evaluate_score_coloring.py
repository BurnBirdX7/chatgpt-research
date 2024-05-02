from __future__ import annotations

from typing import List

import ujson
import logging
import numpy as np
import numpy.typing as npt

import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

from src.score_coloring_pipeline import ScoreColoringPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _score_cover(target: float):
    def score_cover(probs: npt.NDArray[np.float32]) -> float:
        return (probs >= target).sum().astype(float) / float(len(probs))

    target_s = str(target).replace(".", "_")
    score_cover.__name__ = f"score_cover_{target_s}"

    return score_cover


def roc_curve(passages: pd.DataFrame, start_idx: int):
    statistics = [_score_cover(0.01), _score_cover(0.001), _score_cover(0.0001)]

    pipeline = ScoreColoringPipeline("max-score")
    pipeline.store_intermediate_data = False
    pipeline.force_caching("input-tokenized")

    def _max(slice_: List[float]) -> np.float32:
        return np.max(slice_).astype(np.float32)

    pipeline.nodes["scores"].score_func = _max

    with open("progress.json", "r") as f:
        progress = ujson.load(f)

    preds = {f.__name__: [] for f in statistics} | progress["preds"]

    for i, x_test, y_true in passages[start_idx:].itertuples():
        logger.info(f"Evaluating {i + 1} / {len(passages)}...")
        res = pipeline.run(x_test)
        for func in statistics:
            preds[func.__name__].append(func(res.last_node_result))

        with open("progress.json", "w") as f:
            progress["idx"] = i + 1
            progress["preds"] = preds
            ujson.dump(progress, f, indent=2)

    for func, y_pred in preds.items():
        fpr, tpr, thresholds = metrics.roc_curve(passages.drop(progress["skips"])["supported"], y_pred)
        roc_auc = metrics.auc(fpr, tpr)
        plt.title(f"ROC curve for {func} func")
        plt.plot(fpr, tpr, "b", label=f"AUC = {roc_auc:0.2f}")
        plt.legend(loc="lower right")
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.show()


def start():
    passages = pd.read_csv("selected_passages.csv")
    with open("progress.json", "r") as f:
        start_idx = int(ujson.load(f)["idx"])

    logger.info(f"Starting evaluation with progress counter on {start_idx}")
    roc_curve(passages, start_idx)
