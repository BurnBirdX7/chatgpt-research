import os
import sys

import openai
import json

from src import Chat, Dialogue, Question, Config

gpt_model: str = "gpt-3.5-turbo-0301"
quiz_name: str = "pop_quiz_1"

prompt = "I want to ask you quiz questions. Provide simple short answer " \
         "(one sentence, keep under 10 words.) " \
         "Place symbol # after your answer. " \
         "Then provide explanation for the answer (no more than one short paragraph, 200 words max). " \
         "First question: "


def main() -> None:
    # Setup openAI chat
    openai.api_key = os.environ["OPENAI_API_KEY"]

    dialogue = Dialogue()
    dialogue.limit_user_messages = 10
    dialogue.set_system_prompt(prompt)

    chat = Chat(dialogue, gpt_model)
    chat.seconds_to_wait = 30

    # Setup questions
    questions = Question.load_json(Config.artifact(quiz_name + ".json"))
    answers: list[str] = []

    try:
        for i, q in enumerate(questions):
            print(f"{i + 1} / {len(questions)}")

            if q.no_open:
                print("Skipping question...")
                continue

            answer = chat.submit(q.question)
            answers.append(answer)

    except openai.error.OpenAIError as err:
        print("Unexpected error:", file=sys.stderr)
        print(err, file=sys.stderr)

    print("Writing to disk...")
    answers_filename = Config.artifact("answers_" + quiz_name + "_as_open.json")
    json.dump(answers, open(answers_filename, "w"), indent=2)

    chat_filename = Config.artifact("chat_" + quiz_name + "_as_open.json")
    dialogue.dump(chat_filename)


if __name__ == '__main__':
    main()
