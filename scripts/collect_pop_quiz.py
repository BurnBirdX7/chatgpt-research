import os
import sys

import openai
import json

from src import Chat, Dialogue, Question

gpt_model: str = "gpt-3.5-turbo-0301"

prompt = "I want to ask you quiz questions. Chose one answer from a list (print exactly it, finish with a dot). " \
         "Then provide explanation for the answer (no more than one short paragraph, 200 words max). " \
         "First question: "


def main():
    # Setup openAI chat
    openai.api_key = os.environ["OPENAI_API_KEY"]

    dialogue = Dialogue()
    dialogue.limit_user_messages = 10
    dialogue.set_prompt(prompt)

    chat = Chat(dialogue, gpt_model)

    # Setup questions
    questions = Question.load_json("pop_quiz_questions.json")
    answers: list[str] = []

    try:
        for i, q in enumerate(questions):
            print(f"{i + 1} / {len(questions)}")

            answer = chat.submit(str(q))
            answers.append(answer)

    except openai.error.OpenAIError as err:
        print("Unexpected error:", file=sys.stderr)
        print(err, file=sys.stderr)

    print("Writing to disk...")
    json.dump(answers, open(gpt_model + "_answers.json", "w"))


if __name__ == '__main__':
    main()
