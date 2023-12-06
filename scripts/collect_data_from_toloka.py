import pandas as pd


file_wth_results_from_toloka = "../results_from_pool_06-12-2023.tsv"


class TolokaBuilder:
    def __init__(self, file_name):
        self.data_frame = pd.DataFrame()
        self.file_name = file_name

    def collect_data_from_results(self):
        self.data_frame = pd.read_csv(self.file_name, sep='\t')
        self.data_frame = self.data_frame[[
            "INPUT:sentence","INPUT:answer_num", "INPUT:question_num", "INPUT:sentence_num", "OUTPUT:result"
        ]]
        print(self.data_frame.head(10))


def main() -> None:
    toloka_results = TolokaBuilder(file_wth_results_from_toloka)
    toloka_results.collect_data_from_results()


if __name__ == '__main__':
    main()
