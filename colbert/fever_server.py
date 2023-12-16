from __future__ import annotations

from typing import List

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

fever_map = pd.read_csv(os.path.join(root, "collections/fever_map.tsv"), delimiter='\t', header=None)
fever_map.rename(inplace=True, columns={0: 'pid', 1: 'id', 2: "urls"})


def get_urls(pid: int) -> List[str] | None:
    res = fever_map[fever_map["pid"] == pid]
    if len(res) == 0:
        return None

    lst = ast.literal_eval(res.iloc[0]["urls"])
    return lst

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
    passages = [searcher.collection[pid] for pid in pids]
    probs = [math.exp(score) for score in scores]
    probs = [prob / sum(probs) for prob in probs]
    topk = []
    for pid, rank, score, prob in zip(pids, ranks, scores, probs):
        text = searcher.collection[pid]
        d = {'text': text, 'pid': pid, 'rank': rank, 'score': score, 'prob': prob, 'urls': get_urls(pid)}
        topk.append(d)
    topk = list(sorted(topk, key=lambda p: (-1 * p['score'], p['pid'])))
    return {"query" : query, "topk": topk}


@app.route("/api/search", methods=["GET"])
def api_search():
    if request.method == "GET":
        counter["api"] += 1
        print("API request count:", counter["api"])
        return api_search_query(request.args.get("query"), request.args.get("k"))
    else:
        return '', 405


if __name__ == "__main__":
    app.run("0.0.0.0", int(os.getenv("PORT")))
