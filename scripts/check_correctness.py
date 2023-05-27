import json
from typing import List

from src import Config

file: str = "incorrect_answers.json"


def yes(s: str) -> bool:
    return s.strip().lower() == 'y'


def no(s: str) -> bool:
    return s.strip().lower() == 'n'


def main():
    filename = Config.artifact(file)
    answers: List[str] = json.load(open(filename, "r"))

    correct = []
    incorrect = []

    for ans in answers:
        sentences = ans.split('.')
        n = len(sentences)

        for i in range(n):
            if i > 0:
                print("prev:", sentences[i - 1])
            print("CURR:", sentences[i])

            if i < n - 1:
                print("next:", sentences[i + 1])

            inp = ''
            while not yes(inp) and not no(inp):
                inp = input("Is this sentence correct? [y/n]")

            if yes(inp):
                correct.append(sentences[i])
            else:
                incorrect.append(sentences[i])

    json.dump(correct, open("correct.json", "w"), indent=2)
    json.dump(incorrect, open("incorrect.json", "w"), indent=2)


if __name__ == '__main__':
    main()
