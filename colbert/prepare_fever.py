# Creates data file for colbert to read

import pandas as pd


DEFAULT_INPUT_LOC = "collections/fever.jsonl"
DEFAULT_OUTPUT_LOC = "collections/fever.tsv"


def main(source: str, destination: str):
    source_json = pd.read_json(source, lines=True)
    destination_tsv = pd.DataFrame()
    destination_tsv['text'] = source_json['claim']
    destination_tsv.to_csv(destination, sep='\t', header=False)


if __name__ == '__main__':
    main(DEFAULT_INPUT_LOC, DEFAULT_OUTPUT_LOC)
