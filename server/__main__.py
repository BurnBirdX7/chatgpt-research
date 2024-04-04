import datetime
import logging
import time
from functools import lru_cache

from flask import Flask, render_template, request, Response

from scripts.coloring_pipeline import get_coloring_pipeline
from server.render_colored_text import render_colored_text

app = Flask(__name__)

coloring_pipeline = get_coloring_pipeline()
coloring_pipeline.check_prerequisites()
coloring_pipeline.force_caching("input-tokenized")
coloring_pipeline.store_intermediate_data = False

logging.basicConfig(level=logging.DEBUG, format='[%(name)s]:%(levelname)s:%(message)s')

@lru_cache(1)
def color_text(text):

    start = time.time()
    result = coloring_pipeline.run(text)
    seconds = time.time() - start

    print(f"Time taken to run: {datetime.timedelta(seconds=seconds)}")

    print("Statistics")
    for _, stat in result.statistics.items():
        print(str(stat))

    tokens = result.cache["input-tokenized"]
    pos2seq = result.last_node_result
    del result
    return pos2seq, tokens

@app.route("/", methods=['GET'])
def request_page():
    return render_template('root_page.html')


@app.route("/result", methods=['POST'])
def result_page():
    user_input = request.form['user_input']
    pos2chain, tokens = color_text(user_input)
    return Response(render_colored_text(user_input, tokens, pos2chain), mimetype='text/html')


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=4567)
