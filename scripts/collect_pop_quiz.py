import os
import sys

import openai
import json

from src import Chat, Dialogue, Question, Config

gpt_model: str = "gpt-3.5-turbo-0301"

prompt = "I want to ask you quiz questions. Chose one answer from a list (print exactly it)." \
         "Place symbol # after your answer. " \
         "Then provide explanation for the answer (no more than one short paragraph, 200 words max). " \
         "First question: "


def main() -> None:
    # Setup openAI chat
    openai.api_key = os.environ["OPENAI_API_KEY"]

    dialogue = Dialogue()
    dialogue.limit_user_messages = 10
    dialogue.set_prompt(prompt)

    chat = Chat(dialogue, gpt_model)

    # Setup questions
    questions = Question.load_json(Config.artifact("pop_quiz_questions.json"))
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
    answers_filename = Config.artifact(gpt_model + "_answers.json")
    json.dump(answers, open(answers_filename, "w"), indent=2)

    chat_filename = Config.artifact(gpt_model + "_pop_history.json")
    dialogue.dump(chat_filename)


if __name__ == '__main__':
    main()
