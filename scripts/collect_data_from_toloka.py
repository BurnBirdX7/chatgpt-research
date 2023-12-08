import dataclasses
from collections import defaultdict
from typing import List, Dict, Tuple, DefaultDict

import pandas as pd

file_wth_results_from_toloka = "../results_from_pool_06-12-2023.tsv"
file_with_whole_data_from_toloka = "../tasks_from_pool_06-12-2023.tsv"


@dataclasses.dataclass
class Sentence:
    sentence: int
    text: str
    label: str


@dataclasses.dataclass
class ResponseData:
    file: str
    question: int
    answer: int
    label: List[Sentence]


class TolokaBuilder:
    def __init__(self, file_name_result_data, file_name_whole_data):
        self.df_results = None
        self.df_tasks = None
        self.labels = pd.DataFrame({
            "OUTPUT:result": ["True", "False", "Cannot say", "Nonsense"],
            "avg_enum": [1, 0, 0, 0],
        })
        self.empty_data = pd.DataFrame()
        self.result = pd.DataFrame()
        self.filename_results = file_name_result_data
        self.filename_tasks = file_name_whole_data
        self.collect_data_from_results()
        self.collect_data_from_tasks()

    def collect_data_from_results(self):
        self.df_results = pd.read_csv(self.filename_results, sep='\t', index_col=0)
        self.df_results = self.df_results[[
            "INPUT:sentence",
            "INPUT:quiz_file",
            "INPUT:question_num",
            "INPUT:answer_num",
            "INPUT:sentence_num",
            "OUTPUT:result",
        ]]

    def get_not_labeled_data(self):
        first = self.df_tasks
        second = self.df_results.drop_duplicates(
            subset=['INPUT:sentence', 'INPUT:quiz_file', 'INPUT:question_num', 'INPUT:answer_num',
                    'INPUT:sentence_num'])
        merge_df = second.merge(
            first,
            how="outer",
            on=['INPUT:sentence', 'INPUT:quiz_file', 'INPUT:question_num', 'INPUT:answer_num', 'INPUT:sentence_num'],
            indicator=True
        )
        self.empty_data = merge_df[merge_df['_merge'] == 'right_only']
        print(self.empty_data)

    def join_data_frame_result_data_and_labels(self):
        self.df_results = self.df_results.merge(
            self.labels,
            how="left",
            on="OUTPUT:result"
        ).sort_values(by=['INPUT:quiz_file', 'INPUT:question_num', 'INPUT:answer_num', 'INPUT:sentence_num'])
        # self.data_frame_result_data['mean'] = self.data_frame_result_data.loc[:, ["avg_enum"]].mean(axis=1)
        self.df_results = self.df_results.groupby([
            "INPUT:sentence",
            "INPUT:quiz_file",
            "INPUT:question_num",
            "INPUT:answer_num",
            "INPUT:sentence_num",
        ])["avg_enum"].mean().reset_index().sort_values(
            by=['INPUT:quiz_file', 'INPUT:question_num', 'INPUT:answer_num', 'INPUT:sentence_num'])
        self.df_results.loc[self.df_results['avg_enum'] >= 0.5, ['avg_enum']] = True
        self.df_results.loc[self.df_results['avg_enum'] < 0.5, ['avg_enum']] = False

        print(self.df_results)
        print(self.df_results.columns)

    def collect_data_from_tasks(self):
        self.df_tasks = pd.read_csv(self.filename_tasks, sep='\t', index_col=0)
        self.df_tasks = self.df_tasks[[
            "INPUT:sentence",
            "INPUT:quiz_file",
            "INPUT:question_num",
            "INPUT:answer_num",
            "INPUT:sentence_num"
        ]]

    def concatinate_two_dataframes(self):
        self.result = pd.concat([self.df_results, self.empty_data], ignore_index=True)
        self.result.sort_values(by=['INPUT:quiz_file', 'INPUT:question_num', 'INPUT:answer_num', 'INPUT:sentence_num'])
        self.result.drop_duplicates()
        print(self.result)


def main() -> None:
    toloka_results = TolokaBuilder(file_wth_results_from_toloka, file_with_whole_data_from_toloka)
    # toloka_results.collect_data_from_results()
    # toloka_results.collect_data_from_tasks()
    # toloka_results.get_not_labeled_data()
    # toloka_results.join_data_frame_result_data_and_labels()
    # toloka_results.concatinate_two_dataframes()

    # toloka_results.join_data_frame_result_data_and_labels()

    responses_map = {}
    idx = 0
    result:DefaultDict[Tuple[str, int,int], List[Sentence]] = defaultdict(list)
    for index, row in toloka_results.df_tasks.iterrows():
        print(f'{row=}')
        df = toloka_results.df_results
        mask = (df['INPUT:quiz_file'] == row['INPUT:quiz_file']) & \
               (df['INPUT:question_num'] == row['INPUT:question_num']) & \
               (df['INPUT:answer_num'] == row['INPUT:answer_num']) & \
               (df['INPUT:sentence_num'] == row['INPUT:sentence_num'])
        selection=df[mask]
        sentence=Sentence(sentence=row['INPUT:sentence_num'], text=row['INPUT:sentence'], label='')
        if len(selection)==0:
            sentence.label= 'NoData'
        else:
            counts = selection['OUTPUT:result'].value_counts()
            if counts.max() <= counts.sum()/2:
                sentence.label = 'NoData'
            else:
                max_label = counts.idxmax()
                if max_label == 'True':
                    sentence.label = 'True'
                elif max_label == 'False':
                    sentence.label = 'False'
                else:
                    sentence.label = 'NoData'

        result[(row['INPUT:quiz_file'], row['INPUT:question_num'],  row['INPUT:answer_num'])].append(sentence)
    responseData=[]

    for (file,question,answer), value in result.items():
        responseData.append(ResponseData(
            file=file,
            question=question,
            answer=answer,
            label=sorted(value, key=lambda v:v.sentence)
        ))
    for key in responseData:
        print(key)

    # response_data = {
    #     'file': row['INPUT:quiz_file'],
    #     'question_num': row['INPUT:question_num'],
    #     'answer_num': row['INPUT:answer_num'],
    #     'labeled': {'text': row['INPUT:sentence'], 'label': row['avg_enum']}
    # }
    #
    # responses_map[idx] = response_data
    # idx += 1

    # for index, row in toloka_results.df_results.iterrows():
    #     response_data = {
    #         'file': row['INPUT:quiz_file'],
    #         'question_num': row['INPUT:question_num'],
    #         'answer_num': row['INPUT:answer_num'],
    #         'labeled': {'text':row['INPUT:sentence'], 'label':row['avg_enum']}
    #     }
    #
    #     responses_map[idx]=response_data
    #     idx+=1
    # for key, value in responses_map.items():
    #     print(f'{key}:{value}')


if __name__ == '__main__':
    main()
