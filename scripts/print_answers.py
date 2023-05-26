import json
from scripts.collect_pop_quiz import Question
from src import Config

questions_file: str = "open_pop_quiz_question_from_misha_second.json"

as_open = False

answers_file: str
if as_open:
    answers_file = "gpt-3.5-turbo-0301_as_open_answers.json"
else:
    answers_file = "gpt-3.5-turbo-0301_pop_from_misha_2_answers.json"


def main() -> None:
    answers: list[str] = json.load(open(Config.artifact(answers_file), 'r'))
    questions = Question.load_json(Config.artifact(questions_file))
    if as_open:
        questions = list(filter(lambda q: not q.no_open, questions))

    for i, (q, a) in enumerate(zip(questions, answers)):
        print(f"{i + 1} / {len(answers)}")

        if '#' not in a:
            short = "[- NO DELIMITER -]"
            long = a.strip()
        else:
            short, long = a.split('#', 1)
            short = short.strip()
            long = long.strip()
        if q.correct_answer.strip().lower() in short.lower():
            continue

        print(f'correct: {q.correct_answer}')
        print(f'answer : {short}')
        print('explanation:')
        accepted_len = 200
        for j in range(1 + len(long) // accepted_len):
            print(f'\t{long[j * accepted_len : (j + 1) * accepted_len]}')


if __name__ == '__main__':
    main()
