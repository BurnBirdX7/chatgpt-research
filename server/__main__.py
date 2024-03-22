from flask import Flask, render_template, request

from scripts.color_pipeline import get_coloring_pipeline
from server.render_colored_text import render_colored_text

app = Flask(__name__)

coloring_pipeline = get_coloring_pipeline()
coloring_pipeline.check_prerequisites()
coloring_pipeline.force_caching("input-tokenized")

@app.route("/", methods=['GET'])
def request_page():
    return render_template('root_page.html')


@app.route("/result", methods=['POST'])
def result_page():
    user_input = request.form['user_input']

    pos2chain, _, cache = coloring_pipeline.run(user_input)
    tokens = cache["input-tokenized"]

    return render_colored_text(user_input, tokens, pos2chain)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=4567)
