from __future__ import annotations

import re
from typing import List, Tuple, Any, Dict

from flask import Flask, render_template, request
from functools import lru_cache
import math
import os
from dotenv import load_dotenv
import pandas as pd

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher
import ast

from src import SourceMapping

load_dotenv()

ROOT = os.path.dirname(os.path.abspath(__file__))
INDEX_ROOT = os.path.join(ROOT, "wiki/indexes")

app = Flask(__name__)


def init_searchers(dir_path: str) -> List[Tuple[Searcher, SourceMapping]]:
    print(f"Looking for indexes in {dir_path}")
    lst: list[Tuple[Searcher, SourceMapping]] = []
    dirs = [d[0] for d in os.walk(INDEX_ROOT)][1:]
    print(f"Dirs:", dirs)

    # format wiki-{wiki_partition_num}-p{start_page}-p{end_page}_passages_{colbert_partition_num}:
    index_pattern = re.compile(r"^wiki-(\d+)-p(\d+)-p(\d+)_passages_(\d+)$")

    for d in dirs:
        m = index_pattern.match(os.path.basename(d))
        if m:
            g = m.groups()

            s = Searcher(index=d)
            collection_path = s.config.collection.path
            source_path = os.path.join(
                os.path.dirname(collection_path),
                f"wiki-{g[0]}-p{g[1]}-p{g[2]}_sources_{g[3]}.csv"
            )
            mapping = SourceMapping.read_csv(source_path)

            print(f" - Found suitable ColBERT index: {d}")
            print(f"\t > collection path: {collection_path}")
            print(f"\t > source-mapping file path: {source_path}")

            lst.append((s, mapping))

    return lst


searchers: List[Tuple[Searcher, SourceMapping]] = init_searchers(INDEX_ROOT)
counter = {"api_calls": 0}


def search(searcher: Searcher, source_mapping: SourceMapping, query: str, k: int) -> list[dict[str, Any]]:
    pids, ranks, scores = searcher.search(query, k=100)
    pids, ranks, scores = pids[:k], ranks[:k], scores[:k]

    topk_dict: Dict[str, Dict[str, Any]] = dict()

    for pid, rank, score in zip(pids, ranks, scores):
        url, p_start, p_end = source_mapping.get_source_and_interval(pid)

        if url not in topk_dict:
            paragraphs = []
            for paragraph_pid in range(p_start, p_end):
                paragraphs.append(searcher.collection[paragraph_pid])

            topk_dict[url] = {
                'text': "\n".join(paragraphs),
                'source_url': url,
                'score': score
            }
        else:
            old_score = topk_dict[url]['score']
            topk_dict[url]['score'] = max(score, old_score)

    return list(topk_dict.values())



@lru_cache(maxsize=1000000)
def api_search_query(query: str, k: str | None):
    print(f"{query=}")

    if k is None:
        k = 10
    else:
        k = min(int(k), 100)

    topk: List[Dict[str, Any]] = []
    for searcher, sources in searchers:
        topk += search(searcher, sources, query, k)

    topk = list(sorted(topk, key=lambda x: x['score'], reverse=True))
    return {"query" : query, "topk": topk[:100]}


@app.route("/api/search", methods=["GET"])
def api_search():
    if request.method == "GET":
        counter["api"] += 1
        print("API request count:", counter["api_calls"])
        return api_search_query(request.args.get("query"), request.args.get("k"))
    else:
        return '', 405


@app.route("/", methods=["GET"])
def root():
    page = """
    <!DOCTYPE html>
    <html lang="en">
    <body>
    <form action="/api/search" method="get">
    <input type="text" name="query" value="{query}">
    <button type="submit">Search</button>
    </form>
    </html>
    """

    return page


if __name__ == "__main__":
    app.run("0.0.0.0", int(os.getenv("PORT")))
