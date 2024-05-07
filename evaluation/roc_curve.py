import logging
import pathlib
import typing as t

import pandas as pd
import ujson
from matplotlib import pyplot as plt
from sklearn import metrics

from src.pipeline import Pipeline, PipelineResult

logger = logging.getLogger(__name__)


def roc_curve(
    pipeline: Pipeline,
    statistics: t.Sequence[t.Callable[[PipelineResult], float]],
    passages: pd.DataFrame,
    start_idx: int,
    output: pathlib.Path,
):
    with open("progress.json", "r") as f:
        progress = ujson.load(f)

    preds = {f.__name__: [] for f in statistics} | progress["preds"]

    logger.info(f"Evaluating passages, starting at: {start_idx} / {len(passages)}")
    for i, x_test, y_true in passages[start_idx:].itertuples():
        logger.debug(f"Evaluating {i + 1} / {len(passages)}...")
        res = pipeline.run(x_test)
        for func in statistics:
            preds[func.__name__].append(func(res))

        with open("progress.json", "w") as f:
            progress["idx"] = i + 1
            progress["preds"] = preds
            ujson.dump(progress, f, indent=2)

    logger.info("Generating images...")
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

        filepath = output.joinpath(f"{func}.png")
        plt.savefig(filepath)
        plt.close()
        logger.info(f'Saved "{filepath!s}"')
