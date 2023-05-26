import sys
import time
import openai
import json
import string
from typing import Optional


class Dialogue:
    def __init__(self):
        self.limit_user_messages: Optional[int] = None
        self.user_messages: int = 0
        self.start_prompt: str = ""
        self.history: list[dict[str, str]] = []

    def add_msg(self, msg: dict[str, str]):
        self.history.append(msg)

    def add_user_msg(self, text: str):
        if self.user_messages >= self.limit_user_messages:
            self.reset_dialog()

        self.user_messages += 1
        self.history.append({
            "role": "user",
            "content": text
        })

    def set_prompt(self, text: str):
        self.start_prompt = text
        self.reset_dialog()

    def reset_dialog(self):
        self.history = []
        self.add_msg({
            "role": "user",
            "content": self.start_prompt
        })


class Chat:
    def __init__(self, dialogue: Dialogue, model_name: str):
        self.dialogue: Dialogue = dialogue
        self.model_name: str = model_name
        self.seconds_to_wait: int = 20

    def submit(self, text: str) -> str:

        self.dialogue.add_user_msg(text)
        while True:
            try:
                chat_completion = openai.ChatCompletion.create(model=self.model_name,
                                                               messages=self.dialogue.history)
            except openai.error.RateLimitError as err:
                print(err.error, file=sys.stderr)
                print(f"Hit the time limit. Waiting {self.seconds_to_wait}s...")
                time.sleep(self.seconds_to_wait)
                print("Continuing")
                continue

            message = chat_completion.choices[0].message
            self.dialogue.add_msg(message)

            return message.content


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

    def get_dict(self):
        dic = {"question": self.question,
               "answers": self.answers,
               "correct": self.correct_answer}
        return dic

    @staticmethod
    def get_dicts(questions: list["Question"]) -> list[dict[str, str]]:
        return list(map(lambda q: q.get_dict(), questions))
