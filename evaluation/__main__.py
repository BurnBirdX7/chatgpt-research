import json
import os
import signal
import subprocess
import sys
import time

import pandas as pd

from evaluation.evaluate_coloring import start, pipeline, queryNode

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def await_colbert():
    while pipeline.prerequisites_check() is not None:
        logger.info("Sleeping while colbert server is starting up...")
        time.sleep(5)
        logger.info("Woke up")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "bootstrap":
        start()

    if len(sys.argv) == 3 and sys.argv[2] == "init":
        with open('progress.json', 'w') as f:
            json.dump({
                'idx': 0,
                'preds': {}
            }, f)

        all_passages = pd.read_csv("passages.csv")
        pos_passages = all_passages[all_passages["supported"]].sample(100)
        neg_passages = all_passages[~all_passages["supported"]].sample(100)
        passages = pd.concat([pos_passages, neg_passages], ignore_index=True)
        passages.to_csv("selected_passages.csv", index=False)

    counter = 0
    while True:
        counter += 1
        logger.info(f"Starting evaluation cycle, counter = {counter}")
        colbert_server = subprocess.Popen(
            ['conda', 'run', '-n', 'colbert', 'python', '-m', 'colbert_search', 'colbert_server']
        )

        logger.info(f"Colbert server PID: {colbert_server.pid}")

        await_colbert()

        main_proc = subprocess.Popen(['python', '-m', 'evaluation'])
        ret_code = main_proc.wait()
        if ret_code != 0:
            logger.info(f"Colbert server PID: {colbert_server.pid}")

            queryNode.request_kill()

            while True:
                logger.info("Waiting for colbert server to die")
                time.sleep(1)
                if pipeline.prerequisites_check() is not None:
                    break

            logger.info(f"Colbert terminated with {colbert_server.poll()}")
            continue
        break

    colbert_server.kill()
