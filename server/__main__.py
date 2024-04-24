from __future__ import annotations

import logging

from flask import Flask, render_template, request, Response, jsonify

from server.render_colored_text import render_colored_text
from server.statistics_funcs import plot_pos_likelihoods, get_top10_target_chains, get_top10_source_chains
from src.pipeline.pipeline_draw import bytes_draw_pipeline
from src.source_coloring_pipeline import SourceColoringPipeline

import server.source_coloring as source_coloring
import server.score_coloring as score_coloring

# FLASK
app = Flask(__name__)


@app.route("/", methods=["GET"])
def request_page():
    return render_template("root_page.html.j2", resume_points=source_coloring.get_resume_points())


@app.route("/result", methods=["POST"])
def result_html():
    if "type" not in request.form:
        return Response('"type" is required')

    type_ = request.form["type"]

    user_input = request.form["user_input"]
    store_data = "store" in request.form
    if type_ == "score":
        coloring_variants = score_coloring.run(user_input, store_data)
    elif type_ == "source":
        coloring_variants = source_coloring.run(user_input, store_data)
    else:
        return Response(f'Unsupported type "{type_}"')
    return Response(render_colored_text(user_input, coloring_variants), mimetype="text/html")


@app.route("/resume", methods=["GET"])
def resume_html():
    if "type" not in request.args or request.args["type"] not in ["score", "source"]:
        return Response("Missing or unknown type", 400)
    type_ = request.args["type"]

    if "resume_point" not in request.args:
        resume_point = None
    else:
        resume_point = request.args["resume_point"]

    if type_ == "score":
        input_text, coloring_variants = score_coloring.resume(resume_point)
    else:
        input_text, coloring_variants = source_coloring.resume(resume_point)

    return Response(render_colored_text(input_text, coloring_variants), mimetype="text/html")


@app.route("/visualize", methods=["GET"])
def visualize_png():
    img = bytes_draw_pipeline(SourceColoringPipeline.new_extended("coloring"))
    return Response(img, mimetype="image/png")


@app.route("/api/plots/<string:key>/<int:target_pos>", methods=["GET"])
def plot_png(key: str, target_pos: int):
    target_likelihood = request.args.get("likelihood", 0.0, float)
    return Response(plot_pos_likelihoods(target_pos, target_likelihood, key))


@app.route("/api/target-chains/<string:key>/<int:target_pos>", methods=["GET"])
def target_chains_json(key: str, target_pos: int):
    return jsonify(get_top10_target_chains(target_pos, key))


@app.route("/api/source-chains/<string:key>/<path:source_name>/<int:source_pos>")
def source_chains_json(key: str, source_name: str, source_pos: int):
    return jsonify(get_top10_source_chains(key, source_name, source_pos))


if __name__ == "__main__":
    # TODO: Add option to change debug level
    logging.basicConfig(level=logging.DEBUG, format="[%(name)s]:%(levelname)s:%(message)s")
    app.run(host="127.0.0.1", port=4567)
