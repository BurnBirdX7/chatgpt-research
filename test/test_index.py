import statistics

import numpy as np  # type: ignore
import faiss  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from src import Roberta, Config, EmbeddingsBuilder, Index

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


def test_wiki(index: Index, text: str, expected_url: str) -> None:
    embeddings = EmbeddingsBuilder(tokenizer, model, normalize=True).from_text(text)
    result_src, result_dists = index.get_embeddings_source(embeddings)
    dist_sum: float = 0.0
    tp = []
    fn = []

    for i, (src, dist) in enumerate(zip(result_src, result_dists)):
        if dist >= Config.threshold:
            dist_sum += dist
            if src == expected_url:
                tp.append(dist)
        else:
            if src == expected_url:
                fn.append(dist)

    if Config.show_plot:
        tp_mean = statistics.mean(tp)
        fn_mean = statistics.mean(fn)

        if Config.show_plot:
            plt.hist(tp, alpha=0.5, label='TP')
            plt.hist(fn, alpha=0.5, label='FN')
            plt.axvline(tp_mean, color='blue', label=f'TP Mean = {tp_mean:.4f}')
            plt.axvline(fn_mean, color='red', label=f'FN Mean = {fn_mean:.4f}')
            plt.legend()
            plt.xlabel('Cos Distance')
            plt.ylabel('Count')
            plt.show()

    print(
        f"Got expected URL in {len(tp) / len(result_src) * 100:.4f}% of cases, "
        f"average match distance: {dist_sum / len(result_src):.4f}"
    )


def main() -> None:
    read_index: bool = True

    if read_index:
        print("Readings index... ", end='')
        index = Index.load(Config.index_file, Config.mapping_file)
        print("Done")
    else:
        print("Index is being built from wiki... ")
        index = Index.from_config_wiki()  # Obsolete

    print("Test [Data] Searching quotes from the same page:")
    print('"Childhood w references"')
    test_wiki(index, childhood_w_refs, childhood_url)
    print('"Childhood w/o references"')
    test_wiki(index, childhood_wo_refs, childhood_url)
    print('"Legacy"')
    test_wiki(index, legacy, legacy_url)


if __name__ == "__main__":
    main()
