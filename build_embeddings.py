from typing import Dict, List

import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaModel
from text_embedding import input_ids_embedding
import wikipediaapi
from IntervalToSource import IntervalToSource

"""
Script:
Scraps Elvis Presley wiki page and builds embeddings
"""


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
    model_name = 'roberta-base'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name).to(device)

    sections_dict = parse_wiki()
    i2t = IntervalToSource()
    input_ids: List[int] = []

    for title, text in sections_dict.items():
        i2t.append_interval(len(input_ids), title)
        tokens = tokenizer.tokenize(text)
        input_ids += tokenizer.convert_tokens_to_ids(tokens)

    i2t.to_csv('ranges.csv')

    # Should we add <s> </s> tags?
    embeddings = input_ids_embedding(input_ids, model)
    pd.DataFrame(embeddings).to_csv('embeddings.csv', index=False)


if __name__ == '__main__':
    main()
