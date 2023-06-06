# This script collects data by asking ChatGPT questions to the quiz
# Parameter: quiz_name, quiz should be supplied in {artifacts}/{quiz_name}.json
# Env: OPENAI_API_KEY - OpenAI's API key
# Call: python scripts/collect_pop_quiz.py pop_quiz_1
# It will generate {artifacts}/answered_{quiz_name}.json file

import os
import sys

import openai

from src import Chat, Dialogue, Question, Config

gpt_model: str = "gpt-3.5-turbo-0301"

prompt = "I want to ask you quiz questions. Chose one answer from a list (print exactly it)." \
         "Place symbol # after your answer. " \
         "Then provide explanation for the answer (no more than one short paragraph, 200 words max). " \
         "First question: "


def main(quiz_name: str) -> None:
    # Setup OpenAI chat
    openai.api_key = os.environ["OPENAI_API_KEY"]

    dialogue = Dialogue()
    dialogue.limit_user_messages = 10
    dialogue.set_prompt(prompt)

    chat = Chat(dialogue, gpt_model)
    chat.seconds_to_wait = 30

    # Setup questions
    questions = Question.load_json(Config.artifact(quiz_name + ".json"))

    try:
        for i, q in enumerate(questions):
            print(f"{i + 1} / {len(questions)}")

            q.given_answers = chat.multisubmit(str(q), 8)

    except openai.error.OpenAIError as err:
        print("Unexpected error:", file=sys.stderr)
        print(err, file=sys.stderr)

    print("Writing to disk...")
    answers_filename = Config.artifact("answered_" + quiz_name + ".json")
    Question.save_json(questions, answers_filename)

    chat_filename = Config.artifact("chat_" + quiz_name + ".json")
    chat.dialogue.dump(chat_filename)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError("Incorrect number of supplied parameters")
    main(sys.argv[1])
