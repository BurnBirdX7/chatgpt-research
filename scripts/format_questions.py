import json
import sys

from src import Question, Config
import re


def alarm_unicode(text: str):
    for i, t in enumerate(text):
        if ord(t) > 127:
            print(f'Non ASCII character: "{text[i - 10 : i + 10]}" at position {i}', file=sys.stderr)


def format_answer(text: str) -> str:
    return re.sub(r"[\'.]+", "", text.strip())


def format_questions(filename: str):
    questions = Question.load_json(filename)

    for q in questions:
        q.answers = list(map(format_answer, q.answers))
        q.correct_answer = format_answer(q.correct_answer)

    json.dump(Question.get_dicts(questions), open(filename, 'w'), indent=2)


def main():
    format_questions(Config.artifact('pop_quiz_questions.json'))
    format_questions(Config.artifact('open_pop_quiz_question_from_misha.json'))


if __name__ == '__main__':
    main()
