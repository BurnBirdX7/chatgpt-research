import os
import sys
import time

import openai
import json
import string


class Question:
    def __init__(self, question: str, answers: list[str], correct_answer: str):
        self.question = question
        self.answers = answers
        self.correct_answer = correct_answer

    @staticmethod
    def load_json(filename: str) -> list["Question"]:
        f = open(filename, "r")
        j = json.load(f)

        l: list["Question"] = []

        for q in j:
            q_text = q['question']
            q_ans = q['answers']
            q_corr = q['correct']
            l.append(Question(q_text, q_ans, q_corr))

        return l

    def __str__(self) -> str:
        s = f"Q: {self.question}\n"
        for letter, answer in zip(string.ascii_lowercase, self.answers):
            s += f'\t{letter}) {answer}\n'

        return s


gpt_model: str = "gpt-3.5-turbo"

prompt = "I want to ask you quiz questions. Chose one answer from a list (print exactly it, finish with a dot). " \
         "Then provide explanation for the answer (no more than one short paragraph, 200 words max). " \
         "First question: "


def main():
    openai.api_key = os.environ["OPENAI_API_KEY"]

    dialog: list[dict[str, str]] = []
    answers: list[dict[str, str]] = []

    questions = Question.load_json("pop_quiz_questions.json")
    questions[0].question = prompt + questions[0].question

    i = 0
    try:
        while i < len(questions):
            print(f"{i + 1} / {len(questions)}...")

            q = questions[i]
            q_message = {
                "role": "user",
                "content": str(q)
            }

            dialog.append(q_message)

            try:
                char_completion = openai.ChatCompletion.create(model=gpt_model, messages=dialog)
            except openai.error.RateLimitError as err:
                print(err.error, file=sys.stderr)
                print("Hit the time limit. Waiting...")
                time.sleep(20)
                print("Continuing")
                continue

            message = char_completion.choices[0].message
            answers.append(message.content)
            dialog.append(message)

            i += 1
    except object as err:
        print("Unexpected error:", file=sys.stderr)
        print(err, file=sys.stderr)

    print("Writing to disk...")
    json.dump(answers, open(gpt_model + "_answers.json", "w"))


if __name__ == '__main__':
    main()
