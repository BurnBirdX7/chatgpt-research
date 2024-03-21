from flask import Flask, render_template, request

from scripts.color_pipeline import get_coloring_pipeline

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def request_page():
    return render_template('root_page.html')


@app.route("/result", methods=['POST'])
def result_page():
    user_input = request.form['user_input']

    coloring_pipeline = get_coloring_pipeline()

    pos2chain, _ = coloring_pipeline.run(user_input)

    return


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=4567)
