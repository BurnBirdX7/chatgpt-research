from __future__ import annotations

from typing import List, Tuple

from flask import Flask, render_template, request
from functools import lru_cache
import math
import os
from dotenv import load_dotenv
import pandas as pd

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher
import ast

load_dotenv()

root = os.path.dirname(os.path.abspath(__file__))

INDEX_NAME = "fever"
INDEX_ROOT = os.path.join(root, "wiki/indexes")

fever_map = pd.read_csv(os.path.join(root, "collections/fever_map.tsv"), delimiter="\t")


def get_fever_data(pid: int) -> Tuple[bool, List[str]]:
    """
    Retrieve the fever data for the pid
    :param pid: pid of the passage from FEVER dataset
    :return: (bool, List[str]): Bool that indicates if the claim is supported or not, and a list of evidence URL
    """
    row = fever_map.iloc[pid]
    urls = ast.literal_eval(row["urls"])
    is_supported = row["is_supported"]
    return is_supported, urls


app = Flask(__name__)

searcher = Searcher(index=f"{INDEX_ROOT}/{INDEX_NAME}")
counter = {"api": 0}


@lru_cache(maxsize=1000000)
def api_search_query(query, k):
    print(f"{query=}")

    if k is None:
        k = 10

    k = min(int(k), 100)
    pids, ranks, scores = searcher.search(query, k=100)
    pids, ranks, scores = pids[:k], ranks[:k], scores[:k]
    probs = [math.exp(score) for score in scores]
    probs = [prob / sum(probs) for prob in probs]
    topk = []
    for pid, rank, score, prob in zip(pids, ranks, scores, probs):
        text = searcher.collection[pid]
        is_supported, urls = get_fever_data(pid)

        d = {
            "text": text,
            "pid": pid,
            "rank": rank,
            "score": score,
            "prob": prob,
            "urls": urls,
            "is_supported": str(is_supported).lower(),
        }

        topk.append(d)
    topk = list(sorted(topk, key=lambda p: (-1 * p["score"], p["pid"])))
    return {"query": query, "topk": topk}


@app.route("/api/search", methods=["GET"])
def api_search():
    if request.method == "GET":
        counter["api"] += 1
        print("API request count:", counter["api"])
        return api_search_query(request.args.get("query"), request.args.get("k"))
    else:
        return "", 405


if __name__ == "__main__":
    app.run("0.0.0.0", int(os.getenv("PORT")))
