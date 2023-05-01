import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import torch  # type: ignore
import faiss  # type: ignore
import roberta

from text_embedding import text_embedding
from build_index import build_index_from_file
from IntervalToSource import IntervalToSource

import config

"""
Script:
Loads pre-built embeddings to build faiss index and tries to search
"""

__all__ = ["main"]  # Export nothing but main

tokenizer, model = roberta.get_default()

# from 'Childhood in Tupelo' section
childhood_w_refs = (
    "Presley's father Vernon was of German, Scottish, and English origins,[12] and a descendant of the "
    "Harrison family of Virginia through his mother, Minnie Mae Presley (née Hood).[8] Presley's mother "
    "Gladys was Scots-Irish with some French Norman ancestry.[13] She and the rest of the family believed "
    "that her great-great-grandmother, Morning Dove White, was Cherokee.[14][15][16] This belief was restated "
    "by Elvis's granddaughter Riley Keough in 2017.[17] Elaine Dundy, in her biography, supports the belief.["
    "18]"
)

childhood_wo_refs = (
    "Presley's father Vernon was of German, Scottish, and English origins, and a descendant of the "
    "Harrison family of Virginia through his mother, Minnie Mae Presley (née Hood). Presley's mother "
    "Gladys was Scots-Irish with some French Norman ancestry. She and the rest of the family believed "
    "that her great-great-grandmother, Morning Dove White, was Cherokee. This belief was restated "
    "by Elvis's granddaughter Riley Keough in 2017. Elaine Dundy, in her biography, supports the belief."
)
childhood_url = "https://en.wikipedia.org/wiki/Elvis_Presley#Childhood_in_Tupelo"

# from 'Legacy' section
legacy = (
    "President Jimmy Carter remarked on Presley's legacy in 1977: \"His music and his personality, fusing the "
    "styles of white country and black rhythm and blues, permanently changed the face of American popular "
    "culture. His following was immense, and he was a symbol to people the world over of the vitality, "
    'rebelliousness, and good humor of his country."'
)
legacy_url = "https://en.wikipedia.org/wiki/Elvis_Presley#Legacy"


def test_request(index: faiss.Index, q: np.ndarray) -> None:
    k = 4
    dist, ind = index.search(q, k)
    print(f"Distances to {k} nearest neighbours:")
    print(dist)
    print(f"Indexes of {k} nearest neighbours:")
    print(ind)


def test_wiki(index: faiss.Index, text: str, expected_url: str) -> None:
    i2s = IntervalToSource.read_csv(config.ranges_file)

    embeddings = text_embedding(text, tokenizer, model)
    faiss.normalize_L2(embeddings)
    #
    result_dists, result_ids = index.search(embeddings, 1)
    expected_count: int = 0
    dist_sum: float = 0.0
    for i, (token_dists, token_ids) in enumerate(zip(result_dists, result_ids)):
        dist = token_dists[0]
        idx = token_ids[0]
        src = i2s.get_source(idx)

        if src == expected_url:
            expected_count += 1
            dist_sum += dist

    print(
        f"Got expected URL in {expected_count / len(result_dists) * 100:.4f}% of cases, "
        f"average match distance: {dist_sum / len(result_dists):.4f}"
    )


def main() -> None:
    sanity_test: bool = True
    read_from_disk: bool = False

    if read_from_disk:
        print("Readings index... ", end='')
        index = faiss.read_index(config.index_file)
        print("Done")
    else:
        index = build_index_from_file(config.embeddings_file)

    if sanity_test:
        print("Loading embeddings... ", end="")
        data = np.array(pd.read_csv(config.embeddings_file),
                        order="C", dtype=np.float32)
        faiss.normalize_L2(data)
        print("Done")

        print("Searching first 5 embeddings...")
        test_request(index, data[:5])

        print("Searching last 5 embeddings...")
        test_request(index, data[-5:])

    print("Searching quotes from the same page:")
    print('"Childhood w references"')
    test_wiki(index, childhood_w_refs, childhood_url)
    print('"Childhood w/o references"')
    test_wiki(index, childhood_wo_refs, childhood_url)
    print('"Legacy"')
    test_wiki(index, legacy, legacy_url)


if __name__ == "__main__":
    main()
