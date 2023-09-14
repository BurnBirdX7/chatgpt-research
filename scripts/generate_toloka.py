"""
Script takes answered questions and converts them into toloka tasks
Usage: python scripts [directory]
       directory must contain files with answered questions
       directory must be a subdirectory of artifacts dir
       if directory not specified, files will be taken from artifacts
"""

import csv
import os
from datetime import datetime
import sys
import re
from typing import List

import pandas as pd

from src import Question, Config


def table_from_answer(answer: str) -> pd.DataFrame:
    parts = answer.split('##')
    if len(parts) != 2 or len(parts[1].strip()) == 0:
        print(f"Wrong format in answer: {answer}", file=sys.stderr)
        return pd.DataFrame()

    sentences = re.split("[.?!;]", parts[1])
    sentences = list(filter(lambda s: len(s) > 0, sentences))

    n = len(sentences)

    df = pd.DataFrame()

    for num, sent in enumerate(sentences):
        if len(sent) < 15:
            print(f'ALERT: sentence is very short ({num}, "{sent}")')

    df['sentence'] = sentences
    df['previous'] = [None] + sentences[:n - 1]
    df['following'] = sentences[1:] + [None]
    df['sentence_num'] = list(range(n))

    return df


def table_from_question(question: Question) -> pd.DataFrame:
    df_list = []
    for num, answer in enumerate(question.given_answers):
        print(f'-- answer: {num} --')

        answer_df = table_from_answer(answer)
        answer_df['answer_num'] = num
        df_list.append(answer_df)

    return pd.concat(df_list)


def table_from_many_questions(questions: List[Question]) -> pd.DataFrame:
    df_list = []
    for num, question in enumerate(questions):
        print(f'-- question: {num} --')

        quest_df = table_from_question(question)
        quest_df['question_num'] = num
        df_list.append(quest_df)

    return pd.concat(df_list)


def get_files(dirname: str) -> List[str]:
    dirname = Config.artifact(dirname)
    return [
        os.path.join(dirname, f)
        for f in os.listdir(dirname)
        if os.path.isfile(os.path.join(dirname, f)) and
           f.startswith("filtered_") and
           f.endswith(".json")
    ]


def main(dirname: str):
    files = get_files(dirname)

    print("Files:")
    for file in files:
        print(f"\t{file}")

    df_list = []

    for file in files:
        print(f'-- file: {file} --')

        questions = Question.load_json(file)
        df = table_from_many_questions(questions)
        df['quiz_file'] = file
        df_list.append(df)

    df = pd.concat(df_list)

    headers = list(map(lambda col: f'INPUT:{col}', df.columns))
    df.to_csv(Config.artifact(f'toloka_{datetime.now().date()}.tsv'), quoting=csv.QUOTE_MINIMAL, sep='\t', index=False, header=headers)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        main("")
    else:
        main(sys.argv[1])
