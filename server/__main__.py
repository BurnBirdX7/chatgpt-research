from __future__ import annotations

import datetime
import logging
import time
from collections import OrderedDict
from functools import lru_cache
from typing import List, Tuple

from flask import Flask, render_template, request, Response

from scripts.coloring_pipeline import get_coloring_pipeline
from server.render_colored_text import render_colored_text, Coloring
from src import ChainingNode
from src.pipeline import Pipeline


# FLASK
app = Flask(__name__)


# Pipeline configuratio
def pipeline_preset(name: str, use_bidirectional_chaining: bool) -> Pipeline:
    pipeline = get_coloring_pipeline(name)
    pipeline.assert_prerequisites()
    pipeline.force_caching("input-tokenized")
    pipeline.force_caching("$input")
    pipeline.store_optional_data = True
    pipeline.dont_timestamp_history = True
    all_chains: ChainingNode = pipeline.nodes["all-chains"]  # type: ignore
    all_chains.use_bidirectional_chaining = use_bidirectional_chaining
    return pipeline


# Create and configure Pipelines
unidir_pipeline = pipeline_preset("unidirectional", use_bidirectional_chaining=False)
bidir_pipeline = pipeline_preset("bidirectional", use_bidirectional_chaining=True)


@lru_cache(5)
def color_text(
    text: str | None, resume_node: str = "all-chains"
) -> Tuple[str, List[Coloring]]:
    if text is not None and unidir_pipeline.store_optional_data:
        unidir_pipeline.cleanup_file(unidir_pipeline.unstamped_history_filepath)
        bidir_pipeline.cleanup_file(bidir_pipeline.unstamped_history_filepath)

    coloring_variants = []
    start = time.time()

    # UNIDIRECTIONAL
    if text is None:
        result = unidir_pipeline.resume_from_disk(
            unidir_pipeline.unstamped_history_filepath, resume_node
        )
    else:
        result = unidir_pipeline.run(text)
    coloring_variants.append(
        Coloring(
            name="Unidirectional chaining",
            tokens=result.cache["input-tokenized"],
            pos2chain=result.last_node_result,
        )
    )
    stats = OrderedDict()
    stats["unidirectional"] = result.statistics

    # BIDIRECTIONAL
    if text is None:
        result = bidir_pipeline.resume_from_disk(
            bidir_pipeline.unstamped_history_filepath, resume_node
        )
    else:
        result = bidir_pipeline.resume_from_cache(result, "all-chains")

    coloring_variants.append(
        Coloring(
            name="Bidirectional chaining",
            tokens=result.cache["input-tokenized"],
            pos2chain=result.last_node_result,
        )
    )
    stats["bidirectional"] = result.statistics

    # Preserve input
    if text is None:
        text = result.cache["$input"]
    del result

    seconds = time.time() - start

    print(f"Time taken to run: {datetime.timedelta(seconds=seconds)}")
    for i, (name, stats) in enumerate(stats.items()):
        print(f"Statistics (run {i+1}, {name})")
        for _, stat in stats.items():
            print(str(stat))

    return text, coloring_variants


@app.route("/", methods=["GET"])
def request_page():
    return render_template("root_page.html")


@app.route("/result", methods=["POST"])
def result_page():
    user_input = request.form["user_input"]
    store_data = "store" in request.form
    unidir_pipeline.store_intermediate_data = store_data
    bidir_pipeline.store_intermediate_data = store_data

    _, coloring_variants = color_text(user_input)
    return Response(
        render_colored_text(user_input, coloring_variants), mimetype="text/html"
    )


@app.route("/resume", methods=["GET"])
def resume_page():
    if "resume_point" not in request.args:
        resume_point = "all-chains"
    else:
        resume_point = request.args["resume_point"]

    unidir_pipeline.store_intermediate_data = False
    bidir_pipeline.store_intermediate_data = False
    input_text, coloring_variants = color_text(None, resume_node=resume_point)
    return Response(
        render_colored_text(input_text, coloring_variants), mimetype="text/html"
    )


if __name__ == "__main__":
    # TODO: Add option to change debug level
    logging.basicConfig(
        level=logging.DEBUG, format="[%(name)s]:%(levelname)s:%(message)s"
    )
    app.run(host="127.0.0.1", port=4567)
