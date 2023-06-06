# This script filters data collected by `collect_pop_quiz` script, it leaves only questions with incorrect given answers
# Parameter: quiz_name, quiz with GPT's answers should be supplied in {artifacts}/answered_{quiz_name}.json
# Call: python scripts/filter_answers.py pop_quiz_1
# It will generate {artifacts}/filtered_{quiz_name}.json file

import sys
import re
from src import Question, Config


def is_incorrect(answer: str, correct_answer: str, delimiter: str = "#", threshold: float = 0.15):
    short: str
    if "#" not in answer:
        short = answer
    else:
        short, _ = answer.split(delimiter, 1)

    short.strip()
    pattern: str = r"['.,;?/\\\s]"
    words: list[str] = re.split(pattern, short)
    correct_words: list[str] = re.split(pattern, correct_answer)

    n = len(correct_words)
    x = 0
    for w in words:
        if w in correct_words:
            x += 1

    return (x / n) < threshold


def main(quiz_name: str):
    filename = Config.artifact(f"answered_{quiz_name}.json")
    questions = Question.load_json(filename)

    for q in questions:
        q.given_answers = list(filter(lambda ans: is_incorrect(ans, q.correct_answer), q.given_answers))

    questions = list(filter(lambda qu: len(qu.given_answers) > 0, questions))

    out_filename = Config.artifact(f"filtered_{quiz_name}.json")
    Question.save_json(questions, out_filename)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError("Incorrect number of supplied parameters")
    main(sys.argv[1])
