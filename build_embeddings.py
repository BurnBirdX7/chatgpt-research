import numpy as np
import pandas as pd  # type: ignore
import torch  # type: ignore
import wikipediaapi  # type: ignore

from transformers import RobertaTokenizer, RobertaModel  # type: ignore
from typing import Dict, List, Tuple

import roberta
from text_embedding import input_ids_embedding
from IntervalToSource import IntervalToSource

import config

"""
Script:
Scraps wiki pages listed below and builds embeddings

model_name = roberta-base (~3 GB VRAM needed) | roberta-large (~ 5.5 GB VRAM)
"""

page_names = [
    "Elvis_Presley",
    "List_of_songs_recorded_by_Elvis_Presley_on_the_Sun_label",
    "Cultural_impact_of_Elvis_Presley",
    "Cultural_depictions_of_Elvis_Presley",
    "Elvis_has_left_the_building",
    "Elvis_Presley_on_film_and_television",
    "Love_Me_Tender_(film)",
    "Memphis,_Tennessee"
]


def traverse_sections(
    section: wikipediaapi.WikipediaPageSection, page_url: str
) -> Dict[str, str]:
    d = dict()

    # Embed title into paragraph
    text = f" {'=' * section.level} {section.title} {'=' * section.level} \n"
    text += section.text

    url = page_url + "#" + section.title.replace(" ", "_")
    d[url] = text

    for subsection in section.sections:
        d.update(traverse_sections(subsection, page_url))
    return d


def parse_wiki(title: str = "Elvis_Presley") -> Dict[str, str]:
    wikipedia = wikipediaapi.Wikipedia("en")
    target_page = wikipedia.page(title)
    url = target_page.canonicalurl
    d: Dict[str, str] = dict()
    d[url] = target_page.summary

    for section in target_page.sections:
        d.update(traverse_sections(section, url))

    return d


def build_embeddings(tokenizer: RobertaTokenizer, model: RobertaModel) -> Tuple[np.ndarray, IntervalToSource]:
    """
    Computes embeddings
    :param tokenizer: Tokenizer instance
    :param model: Model instance
    :return: Tuple:
                - Embeddings as 2d numpy.array
                - and Interval to Source mapping
    """
    src_map = IntervalToSource()
    embeddings = np.empty((0, model.config.hidden_size))
    sources_dict: Dict[str, str] = dict()

    for i, page in enumerate(page_names):
        print(f"Page {i + 1}/{len(page_names)} in processing")
        sections_dict = parse_wiki(page)
        sources_dict |= sections_dict

        input_ids: List[int] = []
        for title, text in sections_dict.items():
            tokens = tokenizer.tokenize(text)
            input_ids += tokenizer.convert_tokens_to_ids(tokens)
            src_map.append_interval(len(tokens), title)

        page_embeddings = input_ids_embedding(input_ids, model)
        embeddings = np.concatenate([embeddings, page_embeddings])

    return embeddings, src_map


def main() -> None:
    embeddings, mapping = build_embeddings(*roberta.get_default())

    print("Writing to disk... ")
    mapping.to_csv(config.ranges_file)
    pd.DataFrame(embeddings).to_csv(config.embeddings_file, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
