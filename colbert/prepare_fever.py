# Creates data file for colbert to read
from typing import List

import pandas as pd


DEFAULT_INPUT_LOC = "collections/fever.jsonl"
DEFAULT_OUTPUT_COLLECTION_LOC = "collections/fever.tsv"
DEFAULT_OUTPUT_MAPPING_LOC = "collections/fever_map.tsv"


def unpack_sources(pack: List[List[List[str]]]) -> List[str]:
    sources = pack[0]
    sources = map(lambda lst: lst[2], sources)
    sources = map(lambda s: s.replace("-LRB-", "(").replace("-RRB-", ")"), sources)
    sources = map(lambda s: f"en.wikipedia.org/wiki/{s}", sources)
    return list(sources)


def main(source: str, destination: str, destination_map: str):
    print("Preparing FEVER... ", end='')
    source_json = pd.read_json(source, lines=True)
    source_json = source_json[source_json["verifiable"] == "VERIFIABLE"].reset_index(drop=True)

    collection_tsv = pd.DataFrame()
    collection_tsv['text'] = source_json['claim']
    collection_tsv.to_csv(destination, sep='\t', header=False)

    mapping_tsv = pd.DataFrame()
    mapping_tsv['fid'] = source_json['id']
    mapping_tsv['urls'] = source_json['evidence'].map(unpack_sources)
    mapping_tsv['is_supported'] = source_json['label'] == 'SUPPORTS'

    mapping_tsv.to_csv(destination_map, sep='\t')
    print("Done")


if __name__ == '__main__':
    main(DEFAULT_INPUT_LOC, DEFAULT_OUTPUT_COLLECTION_LOC, DEFAULT_OUTPUT_MAPPING_LOC)
