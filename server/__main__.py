from __future__ import annotations

import logging

from flask import Flask, render_template, request, Response

from scripts.coloring_pipeline import get_extended_coloring_pipeline
from server.pipelines import color_text, get_resume_points, plot_pos_likelihoods
from server.render_colored_text import render_colored_text
from src.pipeline.pipeline_draw import bytes_draw_pipeline

# FLASK
app = Flask(__name__)


@app.route("/", methods=["GET"])
def request_page():
    return render_template("root_page.html.j2", resume_points=get_resume_points())


@app.route("/result", methods=["POST"])
def result_page():
    user_input = request.form["user_input"]
    store_data = "store" in request.form
    _, coloring_variants = color_text(user_input, store_data)
    return Response(render_colored_text(user_input, coloring_variants), mimetype="text/html")


@app.route("/resume", methods=["GET"])
def resume_page():
    if "resume_point" not in request.args:
        resume_point = "all-chains"
    else:
        resume_point = request.args["resume_point"]

    input_text, coloring_variants = color_text(None, False, resume_node=resume_point)
    return Response(render_colored_text(input_text, coloring_variants), mimetype="text/html")


@app.route("/visualize", methods=["GET"])
def visualize_img():
    img = bytes_draw_pipeline(get_extended_coloring_pipeline("coloring"))
    return Response(img, mimetype="image/png")


@app.route("/prev/stats", methods=["GET"])
def stats_img():
    target_pos = int(request.args['target_pos'])
    target_likelihood = float(request.args['likelihood'])
    key = request.args['key']
    return Response(plot_pos_likelihoods(target_pos, target_likelihood, key))


if __name__ == "__main__":
    # TODO: Add option to change debug level
    logging.basicConfig(level=logging.DEBUG, format="[%(name)s]:%(levelname)s:%(message)s")
    app.run(host="127.0.0.1", port=4567)
