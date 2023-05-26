import os
import sys

import openai
import json

from src import Chat, Dialogue, Question, Config

gpt_model: str = "gpt-3.5-turbo-0301"

prompt = "I want to ask you quiz questions. Provide simple short answer " \
         "(one sentence, keep under 10 words. Must end with a dot)." \
         "Then provide explanation for the answer (no more than one short paragraph, 200 words max). " \
         "First question: "


def main():
    # Setup openAI chat
    openai.api_key = os.environ["OPENAI_API_KEY"]

    dialogue = Dialogue()
    dialogue.limit_user_messages = 10
    dialogue.set_prompt(prompt)

    chat = Chat(dialogue, gpt_model)
    chat.seconds_to_wait = 30

    # Setup questions
    questions = Question.load_json(Config.artifact("pop_quiz_questions.json"))
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
    filename = Config.artifact(gpt_model + "_as_open_answers.json")
    json.dump(answers, open(filename, "w"), indent=2)


if __name__ == '__main__':
    main()
