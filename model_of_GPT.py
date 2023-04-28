import openai
import api_key

openai.api_key = api_key.key
max_tokens = 128
model_engine = "text-davinci-003"

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

    return completion.choices[0].text
