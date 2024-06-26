from __future__ import annotations

import logging
import signal
import os
import re

import torch
from dotenv import load_dotenv

from typing import List, Tuple, Any, Dict
from flask import Flask, request
from functools import lru_cache
from colbert import Searcher
from src import SourceMapping
from src.config import ColbertServerConfig

load_dotenv()

ROOT = os.path.dirname(os.path.abspath(__file__))
INDEX_ROOT = os.path.join(ROOT, "wiki/indexes")


def init_searchers(dir_path: str) -> List[Tuple[Searcher, SourceMapping]]:
    print(f"Looking for indexes in {dir_path}")
    lst: list[Tuple[Searcher, SourceMapping]] = []
    dirs = [d[0] for d in os.walk(INDEX_ROOT)][1:]
    print("Dirs:", dirs)

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
                f"wiki-{g[0]}-p{g[1]}-p{g[2]}_sources_{g[3]}.csv",
            )
            mapping = SourceMapping.read_csv(source_path)

            print(f" - Found suitable ColBERT index: {d}")
            print(f"\t > collection path: {collection_path}")
            print(f"\t > source-mapping file path: {source_path}")

            lst.append((s, mapping))

    return lst


searchers: List[Tuple[Searcher, SourceMapping]] = []


def search(searcher: Searcher, source_mapping: SourceMapping, query: str, k: int) -> list[dict[str, Any]]:
    pids, ranks, scores = searcher.search(query, k=100)
    pids, ranks, scores = pids[:k], ranks[:k], scores[:k]

    topk_dict: Dict[str, Dict[str, Any]] = dict()

    for pid, rank, score in zip(pids, ranks, scores):
        url, p_start, p_end = source_mapping.get_source_and_interval(pid)

        if url not in topk_dict:
            topk_dict[url] = {
                "text": "\n".join(searcher.collection[paragraph_pid] for paragraph_pid in range(p_start, p_end)),
                "source_url": url,
                "score": score,
            }
        else:
            old_score = topk_dict[url]["score"]
            topk_dict[url]["score"] = max(score, old_score)

    return list(topk_dict.values())


@lru_cache(maxsize=1000000)
def api_search_query(query: str, k_str: str | None):
    print(f"{query=}")

    if k_str is None:
        k = 10
    else:
        k = min(int(k_str), 100)

    topk: List[Dict[str, Any]] = []
    for searcher, sources in searchers:
        topk += search(searcher, sources, query, k)

    topk = list(sorted(topk, key=lambda x: x["score"], reverse=True))
    return {"query": query, "topk": topk[:100]}


def colbert_server(config: ColbertServerConfig):
    app = Flask(__name__)

    @app.route("/", methods=["GET"])
    def root():
        page = """
        <!DOCTYPE html>
        <html lang="en">
        <body>
        <form action="/api/search" method="get">
        <input type="text" name="query" value="Your Query">
        <button type="submit">Search</button>
        </form>
        </html>
        """

        return page

    @app.route(config.api_search_path, methods=["GET"])
    def api_search():
        result = api_search_query(request.args.get("query"), request.args.get("k"))
        torch.cuda.empty_cache()
        return result

    @app.route(config.api_ping_path, methods=["GET"])
    def api_ping():
        return "colbert-pong"

    @app.route(config.api_kill_path, methods=["GET"])
    def api_kill():
        print("REQUESTED SERVER KILL, shutting down...")
        os.kill(os.getpid(), signal.SIGTERM)
        return {"code": "ok", "msg": "Shutting down..."}

    global searchers
    searchers = init_searchers(INDEX_ROOT)
    app.run(config.ip_address, config.port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    colbert_server(ColbertServerConfig.load_from_env())
