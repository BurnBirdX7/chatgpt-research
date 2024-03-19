import collections

from scripts.model_of_GPT import build_page_template
from scripts.build_index_from_potential_sources import build_index_from_potential_sources

from src import EmbeddingsBuilder, Index, Roberta
from typing import Dict, List, Optional

from src.config import EmbeddingsConfig, IndexConfig

# from 'Childhood in Tupelo' section
childhood_w_refs = (
    "Presley's father Vernon was of German, Scottish, and English origins,[12] and a descendant of the "
    "Harrison family of Virginia through his mother, Minnie Mae Presley (née Hood).[8] Presley's mother "
    "Gladys was Scots-Irish with some French Norman ancestry.[13] She and the rest of the family believed "
    "that her great-great-grandmother, Morning Dove White, was Cherokee.[14][15][16] This belief was restated "
    "by Elvis's granddaughter Riley Keough in 2017.[17] Elaine Dundy, in her biography, supports the belief.["
    "18]"
)

childhood_url = "https://en.wikipedia.org/wiki/Elvis_Presley#Childhood_in_Tupelo"


def build_dict_for_color(links: list[str], uniq_color: int) -> Dict[str, str]:
    filtered_links = [link for link in links if link is not None]
    dictionary_of_links = dict(collections.Counter(filtered_links))
    sorted_dict = dict(sorted(dictionary_of_links.items(), key=lambda x: x[1], reverse=True))
    links_with_uniq_colors = dict(list(sorted_dict.items())[:uniq_color])
    uniq_color_dict = {
        'Fuchsia': 'color1',
        'MediumPurple': 'color2',
        'DarkViolet': 'color3',
        'DarkMagenta': 'color4',
        'Indigo': 'color5'
    }

    for link, (_, color_hex) in zip(links_with_uniq_colors, uniq_color_dict.items()):
        links_with_uniq_colors[link] = color_hex

    return links_with_uniq_colors


def prob_test_wiki_with_colored(index: Index, text: str, expected_url: str,
                                uniq_color: int) -> tuple[str, str, str]:
    embeddings = EmbeddingsBuilder(EmbeddingsConfig(normalize=True)).from_text(text)

    result_sources, result_dists = index.get_embeddings_source(embeddings)
    expected_count: int = 0
    dist_sum: float = 0.0

    links: List[Optional[str]] = []

    for i, (dist, source) in enumerate(zip(result_dists, result_sources)):

        if dist < index.config.threshold:
            links.append(None)
        else:
            links.append(source)

        if source == expected_url:
            expected_count += 1
            dist_sum += dist

    print(
        f"Got expected URL in {expected_count / len(result_dists) * 100:.4f}% of cases, "
        f"average match distance: {dist_sum / len(result_dists):.4f}"
    )

    dict_with_uniq_colors = build_dict_for_color(links, uniq_color)

    return build_page_template(text, links, dict_with_uniq_colors)


def color_text(user_input: str) -> tuple[str, str, str]:
    read_index: bool = False

    index: Index
    if read_index:
        print("Readings index... ", end='')
        index = Index.load(IndexConfig())
        print("Done")
    else:
        print("Index is being built from sources ")
        index = build_index_from_potential_sources(user_input)

    print("Test [Data] Searching quotes from the same page:")
    print('"Childhood w references"')
    return prob_test_wiki_with_colored(index, user_input, childhood_url, 5)


# [!] No MAIN
