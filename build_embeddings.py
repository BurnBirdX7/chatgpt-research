from typing import Dict, List

import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaModel
from text_embedding import input_ids_embedding
import wikipediaapi
from IntervalToSource import IntervalToSource

"""
Script:
Scraps wiki pages listed below and builds embeddings

model_name = roberta-base (~3 GB VRAM needed) | roberta-large (~ 5.5 GB VRAM)
"""

model_name = 'roberta-large'
page_names = ['Elvis_Presley',
              'List_of_songs_recorded_by_Elvis_Presley_on_the_Sun_label',
              'Cultural_impact_of_Elvis_Presley',
              'Cultural_depictions_of_Elvis_Presley',
              'Elvis_has_left_the_building',
              'Elvis_Presley_on_film_and_television']


def traverse_sections(section: wikipediaapi.WikipediaPageSection, page_url: str) -> Dict[str, str]:
    d = dict()

    # Embed title into paragraph
    text = f" {'=' * section.level} {section.title} {'=' * section.level} \n"
    text += section.text

    url = page_url + "#" + section.title.replace(' ', '_')
    d[url] = text

    for subsection in section.sections:
        d.update(traverse_sections(subsection, page_url))
    return d


def parse_wiki(title: str = "Elvis_Presley") -> Dict[str, str]:
    wikipedia = wikipediaapi.Wikipedia('en')
    target_page = wikipedia.page(title)
    url = target_page.canonicalurl
    d: Dict[str, str] = dict()
    d[url] = target_page.summary

    for section in target_page.sections:
        d.update(traverse_sections(section, url))

    return d


def main():
    model_name = 'roberta-large'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name).to(device)

    print("Collecting data from wiki... ", end='')
    i2t = IntervalToSource()
    input_ids: List[int] = []
    for page in page_names:
        sections_dict = parse_wiki(page)

        for title, text in sections_dict.items():
            i2t.append_interval(len(input_ids), title)
            tokens = tokenizer.tokenize(text)
            input_ids += tokenizer.convert_tokens_to_ids(tokens)

    i2t.to_csv('ranges.csv')
    print("Done.")

    print("Computing embeddings... ", end='')
    # Should we add <s> </s> tags?
    embeddings = input_ids_embedding(input_ids, model)
    print('Done.\nWriting to disk... ')
    pd.DataFrame(embeddings).to_csv('embeddings.csv', index=False)
    print('Done.')


if __name__ == '__main__':
    main()
