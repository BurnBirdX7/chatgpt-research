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


def init_evaluation(n: int, type: str):
    with open(f".eval_progress.{type}.json", "w") as f:
        json.dump({"idx": 0, "preds": {}, "stats": {}, "p_stats": {}, "skips": [], "type": type}, f, indent=2)

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
        logger.debug(f"ColBERT prereq failed: {check}")
        time.sleep(5)
        check = queryNode.prerequisite_check()


def await_colbert_death():
    while True:
        logger.info("Waiting for colbert server to die...")
        time.sleep(2)
        if queryNode.prerequisite_check() is not None:
            break


def get_progress(action: str) -> dict:
    with open(f".progress.{action}.json", "r") as f:
        return ujson.load(f)


def skip(dat: dict, idx: int):
    assert dat["idx"] == idx
    dat["idx"] += 1
    dat["skips"].append(idx)
    with open("progress.json", "w") as f:
        ujson.dump(dat, f, indent=2)


def main(namespace: argparse.Namespace):
    if not namespace.resume:
        init_evaluation(namespace.sample, namespace.action)

    if not namespace.persistent:
        match namespace.type, namespace.action:
            case "score", "roc":
                evaluation.evaluate_score_coloring.start_roc(namespace.output)
            case "source", "roc":
                evaluation.evaluate_source_coloring.start_roc(namespace.output)
            case "source", "binary":
                evaluation.evaluate_source_coloring.start_bool(namespace.output)
            case "source", "stats":
                evaluation.evaluate_source_coloring.start_stats(namespace.output)
            case _:
                print(f"Unsupported params {namespace.type=}, {namespace.action=}", file=sys.stderr)
        exit(0)

    # Persistent
    if namespace.maintain_colbert:
        fileout: t.TextIO
        if namespace.colbert_stdout == "stdout":
            fileout = sys.stdout
            fileerr = sys.stderr
        else:
            path: pathlib.Path = namespace.colbert_stdout_file
            fileout = open(path, "w")
            fileerr = open(path.parent.joinpath(path.name + "-err"))

    tries: t.Dict[str, int] = defaultdict(lambda: 0)
    while True:
        logger.info(f"Starting evaluation cycle, tries dictionary = {tries!s}")
        if namespace.maintain_colbert:
            colbert_server = subprocess.Popen(
                ["conda", "run", "-n", "colbert", "python", "-m", "colbert_search", "colbert_server"],
                stdout=fileout,
                stderr=fileerr,
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
        if ret_code == 0:
            break

        logger.warning(f"Main process returned {ret_code}, killing ColBERT server...")
        # Execution failed

        # Request colbert server kill:
        if namespace.maintain_colbert:
            queryNode.request_kill()

        # Estimate progress
        progress = get_progress(namespace.action)
        idx = progress["idx"]
        tries[idx] += 1

        # If no progress was made in 5 tries, skip the element
        if tries[idx] >= 3:
            logger.info(f"Skipping idx {idx}")
            skip(progress, idx)

        # Await colbert death
        if namespace.maintain_colbert:
            await_colbert_death()

    if namespace.maintain_colbert:
        queryNode.request_kill()
        await_colbert_death()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("python -m evaluation")
    parser.add_argument("action", choices=["roc", "binary", "stats"], type=str, help="Type of collected statistics")
    parser.add_argument("--type", "-t", choices=["score", "source"], type=str, help="Evaluated method", required=True)
    parser.add_argument(
        "--persistent",
        "-p",
        action="store_true",
        help="Persistent run that relaunches process on error, and maintains ColBERT server",
    )
    parser.add_argument("--resume", "-r", action="store_true", help="Resume previous run")
    parser.add_argument("--output", "-o", type=pathlib.Path, help="Output directory", default=pathlib.Path.cwd())
    parser.add_argument("--sample", "-n", type=int, help="Sample size", default=100)

    parser.add_argument(
        "--colbert-remote",
        dest="maintain_colbert",
        action="store_false",
        help="Rely on remote ColBERT server, see src.config.colbert_server_config for configuration",
    )
    parser.add_argument("--colbert-stdout", choices=["stdout", "file"], default="file")
    parser.add_argument(
        "--colbert-stdout-file",
        type=pathlib.Path,
        help="If --colbert-stdout=stdout, this option is ignored",
        default=".colbert_server_output.txt",
    )

    src.log.add_log_arg(parser)
    namespace = parser.parse_args()
    src.log.process_log_arg(namespace)

    start_time = time.time()
    main(namespace)
    end_time = time.time()

    print(f"Elapsed time: {end_time-start_time:.3f} seconds")
