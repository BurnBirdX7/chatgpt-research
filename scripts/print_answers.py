import json
from scripts.quiz_harvester import Question


def main():
    questions = Question.load_json('pop_quiz_questions.json')
    answers: list[str] = json.load(open('gpt-3.5-turbo_answers.json', 'r'))

    for i, (q, a) in enumerate(zip(questions, answers)):
        a: str
        q: Question

        print(f"{i + 1} / {len(answers)}")

        short, long = a.split('.', 1)
        long = long.strip()

        if q.correct_answer in short:
            continue

        print(f'answer : {short}')
        print(f'correct: {q.correct_answer}')

        print('explanation:')
        accepted_len = 200
        for j in range(len(long) // accepted_len):
            print(f'\t{long[j * accepted_len : (j + 1) * accepted_len]}')


if __name__ == '__main__':
    main()

