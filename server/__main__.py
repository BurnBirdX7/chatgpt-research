from flask import Flask, render_template, request

# from test.color_build_data import main
from scripts.color_text_with_chaining import color_main_with_chaining
from src.config.WikiConfig import WikiConfig
from scripts._elvis_data import elvis_related_articles

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def input_text():
    return render_template('template_of_start_page.html')


@app.route("/result", methods=['POST'])
def result():
    user_input = request.form['user_input']
    file = color_main_with_chaining(user_input, WikiConfig(elvis_related_articles))
    return render_template(file)


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=4567)
