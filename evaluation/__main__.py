import argparse
import json
import subprocess
import time
from collections import defaultdict
from typing import Dict

import pandas as pd
import ujson

from src.config import ColbertServerConfig
from evaluation.evaluate_score_coloring import start

import logging

from src import QueryColbertServerNode

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def init_evaluation():
    with open("progress.json", "w") as f:
        json.dump({"idx": 0, "preds": {}, "skips": []}, f, indent=2)

    all_passages = pd.read_csv("passages.csv")
    pos_passages = all_passages[all_passages["supported"]].sample(500)
    neg_passages = all_passages[~all_passages["supported"]].sample(500)
    passages = pd.concat([pos_passages, neg_passages], ignore_index=True)
    passages.to_csv("selected_passages.csv", index=False)


queryNode = QueryColbertServerNode('-', ColbertServerConfig.load_from_env())

def await_colbert_start():
    while queryNode.prerequisite_check() is not None:
        logger.info("Sleeping while colbert server is starting up...")
        time.sleep(5)


def await_colbert_death():
    while True:
        logger.info("Waiting for colbert server to die...")
        time.sleep(2)
        if queryNode.prerequisite_check() is not None:
            break


def get_progress() -> dict:
    with open("progress.json", "r") as f:
        return ujson.load(f)


def skip(dat: dict, idx: int):
    assert dat["idx"] == idx
    dat["idx"] += 1
    dat["skips"].append(idx)
    with open("progress.json", "w") as f:
        ujson.dump(dat, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("python -m evaluation")
    parser.add_argument('action', choices=['bootstrap', 'init', 'roc'], type=str)
    namespace = parser.parse_args()

    if namespace.action == 'roc':
        start()
        exit(0)

    if namespace.action == 'init':
        init_evaluation()
        exit(0)

    # Bootstrap
    tries: Dict[str, int] = defaultdict(lambda: 0)
    while True:
        logger.info(f"Starting evaluation cycle, counter = {tries!s}")
        colbert_server = subprocess.Popen(
            ["conda", "run", "-n", "colbert", "python", "-m", "colbert_search", "colbert_server"]
        )

        logger.info(f"Colbert server PID: {colbert_server.pid}")

        await_colbert_start()

        main_proc = subprocess.Popen(["python", "-m", "evaluation", "roc"])
        ret_code = main_proc.wait()
        if ret_code != 0:
            logger.warning(f"Main process returned {ret_code}, killing ColBERT server...")
            # Execution failed

            # Request colbert server kill:
            queryNode.request_kill()

            # Estimate progress
            progress = get_progress()
            idx = progress["idx"]
            tries[idx] += 1

            # If no progress was made in 5 tries, skip the element
            if tries[idx] >= 3:
                logger.info(f"Skipping idx {idx}")
                skip(progress, idx)

            # Await colbert death
            await_colbert_death()

            continue
        break

    colbert_server.kill()
