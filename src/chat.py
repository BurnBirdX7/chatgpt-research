import copy
import sys
import time
import openai
import json
import string
from typing import Optional, List, Dict

from progress.bar import Bar, Progress  # type: ignore


class Dialogue:
    def __init__(self: "Dialogue"):
        self.limit_user_messages: Optional[int] = None
        self.user_messages: int = 0
        self.system_prompt: str = ""
        self.history: List[Dict[str, str]] = []
        self.old_history: List[Dict[str, str]] = []

    def add_msg(self, msg: Dict[str, str]):
        self.history.append(msg)

    def add_user_msg(self, text: str):
        if self.user_messages == 0 or (self.limit_user_messages and self.user_messages >= self.limit_user_messages):
            self.reset_dialog()

        self.user_messages += 1
        self.history.append({
            "role": "user",
            "content": text
        })

    def set_system_prompt(self, text: str):
        self.system_prompt = text
        self.reset_dialog()

    def reset_dialog(self):
        self.old_history += self.history
        self.history = []
        self.add_msg({
            "role": "system",
            "content": self.system_prompt
        })

    def dump(self, filename: str):
        all_history = self.old_history + self.history
        json.dump(all_history, open(filename, 'w'), indent=2)


class Chat:
    def __init__(self: "Chat", dialogue: Dialogue, model_name: str):
        self.dialogue: Dialogue = dialogue
        self.model_name: str = model_name
        self.seconds_to_wait: int = 20
        self.suppress_output: bool = False

    def submit(self, text: str, bar: Optional[Progress] = None) -> str:
        reduced_length: bool = False
        timed_out: int = 0
        max_timeout: int = 5  # ------------------ configurable -------------
        self.dialogue.add_user_msg(text)
        while True:
            try:
                chat_completion = openai.ChatCompletion.create(model=self.model_name,
                                                               messages=self.dialogue.history,
                                                               temperature=0.7)

            except openai.error.Timeout as err:
                if timed_out < max_timeout:
                    timed_out += 1
                    if not self.suppress_output:
                        print(err.error, file=sys.stderr)
                        print(f"Connection timed out... Trying again ({timed_out}/{max_timeout})")
                    continue

                if not self.suppress_output:
                    print(err.error, file=sys.stderr)
                    print("Connection timed out again")

            except openai.error.InvalidRequestError as err:
                # Assume that this is a "context is too long error"
                if "context length" in str(err.error) and not reduced_length:
                    self.dialogue.reset_dialog()
                    self.dialogue.add_user_msg(text)
                    reduced_length = True
                elif reduced_length and not self.suppress_output:
                    print(f"Error text: {err.error}", file=sys.stderr)
                    print(f"Context reduction did not resolve the problem. Stopping...")
                    break

                if not self.suppress_output:
                    print(f"Error text: {err.error}", file=sys.stderr)
                    print(f"[assumed] Hit context length limit. Context was reduced. Retrying...")

                continue

            except openai.error.RateLimitError as err:
                if not self.suppress_output:
                    print(f"Error text: {err.error}", file=sys.stderr)
                    print(f"Hit the time limit. Waiting {self.seconds_to_wait}s...")

                if bar is not None:
                    bar.message = "Waiting"
                    bar.next(0)

                time.sleep(self.seconds_to_wait)

                if bar is not None:
                    bar.message = "Submitting"
                    bar.next(0)

                if not self.suppress_output:
                    print("Continuing")
                continue

            message = dict(chat_completion.choices[0].message)
            self.dialogue.add_msg(message)

            return message["content"]
        raise OSError("Unreachable code")

    def multisubmit(self, text: str, resubmission_rate: int = 2):
        dialogue: Dialogue
        answers: List[str] = []

        if resubmission_rate < 1:
            raise ValueError("Invalid resubmission rate")

        bar = Bar("Submitting", max=resubmission_rate)
        for i in range(resubmission_rate):
            bar.next()
            dialogue = copy.deepcopy(self.dialogue)
            chat = Chat(dialogue, self.model_name)
            chat.suppress_output = True
            chat.seconds_to_wait = self.seconds_to_wait
            answers.append(chat.submit(text, bar=bar))
        bar.finish()

        print()
        self.dialogue = dialogue
        return answers


class Question:
    def __init__(self, question: str, answers: List[str], correct_answer: str):
        self.question: str = question
        self.answers: List[str] = answers
        self.correct_answer: str = correct_answer
        self.source: Optional[str] = None
        self.no_open: bool = False
        self.given_answers: List[str] = []

    @staticmethod
    def load_json(filename: str) -> List["Question"]:
        f = open(filename, "r")
        j = json.load(f)

        l: List["Question"] = []

        for q in j:
            q_text = q['question']
            q_ans = q['answers']
            q_corr = q['correct']
            question = Question(q_text, q_ans, q_corr)
            if 'source' in q:
                question.source = q['source']
            if 'no-open' in q:
                question.no_open = q['no-open']
            if 'given-answers' in q:
                question.given_answers = q['given-answers']
            l.append(question)

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

        if self.source is not None:
            dic["source"] = self.source

        if self.no_open:
            dic['no-open'] = True

        if len(self.given_answers) > 0:
            dic['given-answers'] = self.given_answers

        return dic

    @staticmethod
    def get_dicts(questions: List["Question"]) -> List[Dict[str, str]]:
        return list(map(lambda q: q.get_dict(), questions))

    @staticmethod
    def save_json(question_list: List["Question"], filename: str) -> None:
        json.dump(Question.get_dicts(question_list), open(filename, 'w'), indent=2)
