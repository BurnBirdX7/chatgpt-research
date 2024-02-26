from flask import Flask, render_template, request

# from test.color_build_data import main
from scripts.color_text_with_chaining import color_main_with_chaining

app = Flask(__name__)


@app.route("/", methods=['GET', 'POST'])
def input_text():
    if request.method == 'POST':
        user_input = request.form['user_input']
        color_main_with_chaining(user_input)
        return render_template('template_of_result_page.html')
    return render_template('template_of_start_page.html')


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=4567)
