import openai
import api_key
from jinja2 import Template

openai.api_key = api_key.key
max_tokens = 128
model_engine = "text-davinci-003"

template_string = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Result</title>
    <style>
        {{ color }}
    </style>
</head>
<body>
<h1>Result of research</h1>
<a>{{ result }}</a>
</body>
</html>
"""

"""
Creates a model based on Chat-GPT for the answering users questions.
"""


def model(message_from_user):
    completion = openai.Completion.create(
        engine=model_engine,
        prompt=message_from_user,
        max_tokens=max_tokens,
        temperature=0.5,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    build_template(completion)
    return


def build_template(completion):
    template = Template(template_string)
    result_html = template.render(result=completion.choices[0].text, )

    with open("template.html", "w", encoding="utf-8") as f:
        f.write(result_html)


def build_template_simple(completion):
    template = Template(template_string)
    result_html = template.render(result=completion, color=".color1 {color: red;}")

    with open("template.html", "w", encoding="utf-8") as f:
        f.write(result_html)


if __name__ == "__main__":
    # model("что такое python?")
    build_template_simple('<a href="source-url" class="color1">token</a>')
