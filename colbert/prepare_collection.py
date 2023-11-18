# Creates data file for colbert to read
# Input: <arg> path to JSONL file

import sys
import pandas as pd


DEFAULT_INPUT_LOC = "collections/train.jsonl"


def main(source: str, destination: str):
    source_json = pd.read_json(source, lines=True)
    destination_tsv = pd.DataFrame()
    destination_tsv['text'] = source_json['claim']
    destination_tsv.to_csv(destination, sep='\t', header=False)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        print(f"Using default location <{DEFAULT_INPUT_LOC}>")
        main(DEFAULT_INPUT_LOC, sys.argv[1])
    elif len(sys.argv) == 3:
        main(sys.argv[2], sys.argv[1])
    else:
        print("Usage: prepare_collection <destination path> [source path]")
