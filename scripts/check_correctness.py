# This script prompts user to check correctness of given answers
# Parameter: quiz_name; quiz with GPT's filtered answers should be supplied in {artifacts}/filtered_{quiz_name}.json
# Call: python scripts/check_correctness.py pop_quiz_1
# It will generate {artifacts}/surveyed_{quiz_name}.json file

import sys
from typing import List, Dict

from src import Config, Question


def yes(s: str) -> bool:
    return s.strip().lower() == 'y'


def no(s: str) -> bool:
    return s.strip().lower() == 'n'


def survey(sentences: List[str]) -> Dict[str, bool]:
    sentence_correctness: Dict[str, bool] = {}
    sentences = list(
        filter(lambda s: len(s) > 0,
               map(lambda s: s.strip(), sentences)))

    n = len(sentences)
    for i in range(n):
        if i > 0:
            print("prev:", sentences[i - 1])

        print("CURR:", sentences[i].strip())

        if i < n - 1:
            print("next:", sentences[i + 1])

        inp = ''
        while not yes(inp) and not no(inp):
            inp = input("Is this sentence correct? [y/n]")

        sentence_correctness[sentences[i].strip()] = yes(inp)

    return sentence_correctness


def main(quiz_name: str, delimiter: str = "#"):
    filename = Config.artifact(f"filtered_{quiz_name}.json")
    questions = Question.load_json(filename)

    for q in questions:
        answer_correctness: List[Dict[str, bool]] = []

        for answer in q.given_answers:
            if '#' not in answer:
                explanation = answer
            else:
                _, explanation = answer.split(delimiter, 1)

            sentences = explanation.split('.')
            answer_correctness.append(survey(sentences))

        q.given_answers = answer_correctness  # replace list with dictionary

    out_filename = Config.artifact(f"surveyed_{quiz_name}.json")
    Question.save_json(questions, out_filename)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError("Incorrect number of supplied parameters")
    main(sys.argv[1])
