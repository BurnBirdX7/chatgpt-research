"""
Prepares data for colbert and for testing of the whole system
"""

import random
import sys
import time
from typing import List

import numpy as np
import pandas as pd


from src.online_wiki import OnlineWiki
from src import SourceMapping

fever_path = "./colbert_search/collections/fever.jsonl"

def unpack_sources(pack: List[List[List[str]]]) -> List[str]:
    sources = [evidence for sublist in pack for evidence in sublist]
    sources = map(lambda lst: lst[2], sources)
    sources = map(lambda s: s.replace("-LRB-", "(").replace("-RRB-", ")").replace('-COLON-', ':'), sources)
    return list(sources)

def get_sources(lst: list[str]) -> dict[str, str]:
    d = {}
    print(f"expected: {len(lst)}")

    req_time = time.time()
    for i, source in enumerate(lst):
        print('.', end='')
        if (i + 1) % 10 == 0:
            print(i + 1, end='')
        if (i + 1) % 100 == 0:
            print()

        for try_num in range(5):
            try:
                # Add small pause before each request
                to_slp = 0.1 - (time.time() - req_time)
                if to_slp > 0.0:
                    time.sleep(to_slp)

                d |= OnlineWiki.get_sections(source)
                req_time = time.time()
                break

            except ConnectionError as e:
                print(f"ConnectionError: {e}")
                print(f"Retrying ({try_num}) in 1 second...")
                time.sleep(1)

            except Exception as e:
                print(f"Unknown error \"{source}\", {e}", file=sys.stderr)
                print(f"Retrying ({try_num}) in 1 second...")
                time.sleep(1)

    print(" Done")
    return d


def compress(series: pd.Series) -> list[str]:
    return np.unique(np.concatenate(series.map(unpack_sources).to_numpy()))  # type: ignore


def save_sources(data: dict[str, str], passages_file, sources_file) -> None:
    pid = 0
    mapping = SourceMapping()
    with open(passages_file, 'w') as f:
        for url, text in data.items():
            parts = text.split('\n')
            for part in parts:
                passage = part.strip().replace('\t', ' ')
                if len(passage) == 0:
                    continue

                f.write(f"{pid}\t{passage}\n")
                mapping.append_interval(1, url)
                pid += 1

    mapping.to_csv(sources_file)


def fever_process_dataset():
    with open(fever_path, 'r') as f:
        obj = pd.read_json(f, lines=True)

    mask_verifiable_only = obj["verifiable"] == "VERIFIABLE"
    obj = obj[mask_verifiable_only]

    supported = obj[obj["label"] == "SUPPORTS"]
    not_supported = obj[obj["label"] == "REFUTES"]

    # Data:
    sup_passages = pd.DataFrame({
        'passage': supported["claim"],
        'supported': True
    })

    not_sup_passage = pd.DataFrame({
        'passage': not_supported["claim"],
        'supported': False
    })

    # Passages for testing:
    passages = pd.concat([sup_passages, not_sup_passage], ignore_index=True)
    passages.to_csv("passages.csv", index=False)

    # Formulate content for __ideal__ source index:
    # Get titles:
    supporting_sources = compress(supported["evidence"])
    not_supporting_sources = compress(not_supported["evidence"])
    print(len(supporting_sources), len(not_supporting_sources))

    # Get contents:
    print("Query supporting sources")
    supporting_dict = get_sources(random.choices(supporting_sources, k=1000))

    print("Query not-supporting sources")
    not_supporting_dict = get_sources(random.choices(not_supporting_sources, k=1000))

    # Save to disk
    save_sources(supporting_dict, 'wiki-p0-p0_passages_1.tsv', 'wiki-p0-p0_sources_1.csv')
    save_sources(supporting_dict, 'wiki-p0-p0_passages_2.tsv', 'wiki-p0-p0_sources_2.csv')


if __name__ == '__main__':
    fever_process_dataset()
