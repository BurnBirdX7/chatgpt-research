import json
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict
from typing import Dict

import pandas as pd
import ujson

from evaluation.evaluate_source_coloring import start, pipeline, queryNode

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_evaluation():
    with open("progress.json", "w") as f:
        json.dump({"idx": 0, "preds": {}, "skips": []}, f, indent=2)

    all_passages = pd.read_csv("passages.csv")
    pos_passages = all_passages[all_passages["supported"]].sample(100)
    neg_passages = all_passages[~all_passages["supported"]].sample(100)
    passages = pd.concat([pos_passages, neg_passages], ignore_index=True)
    passages.to_csv("selected_passages.csv", index=False)


def await_colbert_start():
    while pipeline.prerequisites_check() is not None:
        logger.info("Sleeping while colbert server is starting up...")
        time.sleep(5)


def await_colbert_death():
    while True:
        logger.info("Waiting for colbert server to die...")
        time.sleep(2)
        if pipeline.prerequisites_check() is not None:
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
    if len(sys.argv) < 2 or sys.argv[1] != "bootstrap":
        start()

    if len(sys.argv) == 3 and sys.argv[2] == "init":
        init_evaluation()
        exit(0)

    tries: Dict[str, int] = defaultdict(lambda: 0)
    while True:
        logger.info(f"Starting evaluation cycle, counter = {tries}")
        colbert_server = subprocess.Popen(
            ["conda", "run", "-n", "colbert", "python", "-m", "colbert_search", "colbert_server"]
        )

        logger.info(f"Colbert server PID: {colbert_server.pid}")

        await_colbert_start()

        main_proc = subprocess.Popen(["python", "-m", "evaluation"])
        ret_code = main_proc.wait()
        if ret_code != 0:
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
