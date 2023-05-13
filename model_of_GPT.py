import openai
import api_key
from jinja2 import Template
from transformers import RobertaTokenizer
import config

openai.api_key = api_key.key
max_tokens = 128
model_engine = "text-davinci-003"

page_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Result</title>
    <link rel="stylesheet" type="text/css" href="style.css">
</head>
<body>
<h1>Result of research</h1>
<pre> {{ gpt_response }} </pre>
{{ result }}
</body>
</html>
"""

link_template = "<a href=\"{{ link }}\" class=\"{{ color }}\">{{ token }}</a>"

def model(message_from_user: str):
    """
    Creates a model based on Chat-GPT for the answering users questions.
    """
    completion = openai.Completion.create(
        engine=model_engine,
        prompt=message_from_user,
        max_tokens=max_tokens,
        temperature=0.5,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    build_page_template(completion.choices[0].text)
    return


def build_list_of_tokens_input(text: str) -> list[str]:
    tokenizer = RobertaTokenizer.from_pretrained(config.model_name)
    tokens = tokenizer.tokenize(text)
    print("tokens", tokens)
    return tokens


def build_link_template(tokens: list[str]) -> str:
    template = Template(link_template)
    tokens = map(lambda s: s.replace('Ġ', ' ').replace('Ċ', '<br/>'), tokens)
    link = "link_1"
    color = "color4"
    output = ""

    for key in tokens:
        output += template.render(link=link, color=color, token=key)

    return output


def build_page_template(completion: str) -> None:
    template = Template(page_template)

    tokens_from_output = build_list_of_tokens_input(completion)
    result_of_color = build_link_template(tokens_from_output)
    result_html = template.render(result=result_of_color, gpt_response=completion)

    with open("output/result.html", "w", encoding="utf-8") as f:
        f.write(result_html)


if __name__ == "__main__":
    model("when was born Elvis?")
