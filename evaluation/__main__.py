import argparse
import json
import pathlib
import subprocess
import sys
import time
from collections import defaultdict
import typing as t

import pandas as pd
import ujson

import evaluation.evaluate_score_coloring
import evaluation.evaluate_source_coloring
import src.log
from src.config import ColbertServerConfig

import logging

from src import QueryColbertServerNode

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def init_evaluation(n: int):
    with open("progress.json", "w") as f:
        json.dump({"idx": 0, "preds": {}, "skips": []}, f, indent=2)

    all_passages = pd.read_csv("passages.csv")
    pos_passages = all_passages[all_passages["supported"]].sample(n // 2)
    neg_passages = all_passages[~all_passages["supported"]].sample(n // 2)
    passages = pd.concat([pos_passages, neg_passages], ignore_index=True)
    passages.to_csv("selected_passages.csv", index=False)


queryNode = QueryColbertServerNode("-", ColbertServerConfig.load_from_env())


def await_colbert_start():
    check = queryNode.prerequisite_check()
    while check is not None:
        logger.info("Sleeping while colbert server is starting up...")
        logger.error(f"ColBERT prereq failed: {check}")
        time.sleep(5)
        check = queryNode.prerequisite_check()


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


def main(namespace: argparse.Namespace):
    if not namespace.resume:
        init_evaluation(namespace.sample)

    if not namespace.persistent:
        # Assume ROC # TODO
        match namespace.type:
            case "score":
                evaluation.evaluate_score_coloring.start(namespace.output)
            case "source":
                evaluation.evaluate_source_coloring.start(namespace.output)
        exit(0)

    # Persistent
    tries: t.Dict[str, int] = defaultdict(lambda: 0)
    while True:
        logger.info(f"Starting evaluation cycle, tries dictionary = {tries!s}")
        if namespace.maintain_colbert:
            colbert_server = subprocess.Popen(
                ["conda", "run", "-n", "colbert", "python", "-m", "colbert_search", "colbert_server"], stdout=sys.stdout
            )

        logger.info(f"Colbert server PID: {colbert_server.pid}")

        await_colbert_start()

        main_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "evaluation",
                namespace.action,
                "--type",
                namespace.type,
                "--resume",
                "--output",
                str(namespace.output),
            ]
        )
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
            if namespace.maintain_colbert:
                await_colbert_death()

            continue
        break

    if namespace.maintain_colbert:
        colbert_server.kill()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("python -m evaluation")
    parser.add_argument("action", choices=["roc"], type=str, help="Type of collected statistics")
    parser.add_argument("--type", "-t", choices=["score", "source"], type=str, help="Evaluated method", required=True)
    parser.add_argument(
        "--persistent",
        "-p",
        action="store_true",
        help="Persistent run that relaunches process on error, and maintains ColBERT server",
    )
    parser.add_argument(
        "--separate-colbert",
        "--remote-colbert",
        dest="maintain_colbert",
        action="store_false",
        help="Rely on remote ColBERT server, see src.config.colbert_server_config for configuration",
    )
    parser.add_argument("--resume", "-r", action="store_true", help="Resume previous run")
    parser.add_argument("--output", "-o", type=pathlib.Path, help="Output directory", default=pathlib.Path.cwd())
    parser.add_argument("--sample", "-n", type=int, help="Sample size", default=100)
    src.log.add_log_arg(parser)
    namespace = parser.parse_args()
    src.log.process_log_arg(namespace)

    main(namespace)
