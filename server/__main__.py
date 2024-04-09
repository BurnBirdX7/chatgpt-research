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

app = Flask(__name__)

coloring_pipeline = get_coloring_pipeline()
coloring_pipeline.assert_prerequisites()
coloring_pipeline.force_caching("input-tokenized")
coloring_pipeline.store_optional_data = True
coloring_pipeline.dont_timestamp_history = True

logging.basicConfig(level=logging.DEBUG, format='[%(name)s]:%(levelname)s:%(message)s')

@lru_cache(5)
def color_text(text: str | None) -> Tuple[str, List[Coloring]]:
    coloring_variants = []
    start = time.time()

    all_chain_node: ChainingNode = coloring_pipeline.nodes["all-chains"]  # type: ignore

    # UNIDIRECTIONAL
    all_chain_node.use_bidirectional_chaining = False
    if text is None:
        result = coloring_pipeline.resume_from_disk(coloring_pipeline.unstamped_history_filepath,
                                                    "all-chains")
    else:
        result = coloring_pipeline.run(text)
    coloring_variants.append(Coloring(name="Unidirectional chaining",
                                      tokens=result.cache['input-tokenized'],
                                      pos2chain=result.last_node_result))
    stats = OrderedDict()
    stats['unidirecinal'] = result.statistics

    # BIDIRECTIONAL
    coloring_pipeline.store_intermediate_data = False
    all_chain_node.use_bidirectional_chaining = True
    result = coloring_pipeline.resume_from_cache(result, "all-chains")
    coloring_variants.append(Coloring(name="Unidirectional chaining",
                                      tokens=result.cache['input-tokenized'],
                                      pos2chain=result.last_node_result))
    stats['bidirectional'] = result.statistics

    # Preserve input
    if text is None and '$input' in result.cache:
        text = result.cache['$input']
    del result

    seconds = time.time() - start

    print(f"Time taken to run: {datetime.timedelta(seconds=seconds)}")
    for i, (name, stats) in enumerate(stats.items()):
        print(f"Statistics (run {i+1}, {name})")
        for _, stat in stats.items():
            print(str(stat))

    return text, coloring_variants


@app.route("/", methods=['GET'])
def request_page():
    return render_template('root_page.html')


@app.route("/result", methods=['POST'])
def result_page():
    user_input = request.form['user_input']
    store_data = 'store' in request.form
    coloring_pipeline.store_intermediate_data = store_data
    if store_data:
        coloring_pipeline.cleanup_file(coloring_pipeline.unstamped_history_filepath)

    _, coloring_variants = color_text(user_input)
    return Response(render_colored_text(user_input, coloring_variants), mimetype='text/html')


@app.route("/resume", methods=['GET'])
def resume_page():
    coloring_pipeline.store_intermediate_data = False
    input_text, coloring_variants = color_text(None)
    return Response(render_colored_text(input_text, coloring_variants), mimetype='text/html')


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=4567)
