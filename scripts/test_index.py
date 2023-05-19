import numpy as np  # type: ignore
import faiss  # type: ignore

from src import Roberta, Config, SourceMapping
from src.embeddings import text_embedding
from .build_index import build_index

"""
Script:
Loads pre-built embeddings to build faiss index and tries to search
"""

__all__ = ["main"]  # Export nothing but main

tokenizer, model = Roberta.get_default()

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


def test_wiki(index: faiss.Index, src_map: SourceMapping, text: str, expected_url: str) -> None:
    embeddings = text_embedding(text, tokenizer, model)
    faiss.normalize_L2(embeddings)
    #
    result_dists, result_ids = index.search(embeddings, 1)
    expected_count: int = 0
    dist_sum: float = 0.0
    for i, (token_dists, token_ids) in enumerate(zip(result_dists, result_ids)):
        dist = token_dists[0]
        idx = token_ids[0]
        src = src_map.get_source(idx)

        if src == expected_url:
            expected_count += 1
            dist_sum += dist

    print(
        f"Got expected URL in {expected_count / len(result_dists) * 100:.4f}% of cases, "
        f"average match distance: {dist_sum / len(result_dists):.4f}"
    )


def main() -> None:
    read_index: bool = True

    if read_index:
        print("Readings index... ", end='')
        index = faiss.read_index(Config.index_file)
        mapping = SourceMapping.read_csv(Config.ranges_file)
        print("Done")
    else:
        print("Index is being built from wiki... ")
        index, mapping = build_index()

    print("Test [Data] Searching quotes from the same page:")
    print('"Childhood w references"')
    test_wiki(index, mapping, childhood_w_refs, childhood_url)
    print('"Childhood w/o references"')
    test_wiki(index, mapping, childhood_wo_refs, childhood_url)
    print('"Legacy"')
    test_wiki(index, mapping, legacy, legacy_url)


if __name__ == "__main__":
    main()
