from typing import Dict, List

import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaModel
import wikipediaapi


wikipedia = wikipediaapi.Wikipedia('en')


"""
This code graps Elvis Presley page from Wikipedia and weeds in to RoBERTa to get token embeddings
Prints acquired data into embeddings.txt file
"""


def traverse_sections(section: wikipediaapi.WikipediaPageSection) -> Dict[str, str]:
    d = dict()

    # Embed title into paragraph
    text = f" {'=' * section.level} {section.title} {'=' * section.level} \n"
    text += section.text

    d[section.title] = text

    for subsection in section.sections:
        d.update(traverse_sections(subsection))
    return d


def parse_wiki(title: str = "Elvis_Presley") -> Dict[str, str]:
    target_page = wikipedia.page(title)
    d: Dict[str, str] = dict()

    for section in target_page.sections:
        d.update(traverse_sections(section))

    return d


"""
Prints acquired data into embeddings.txt file
"""


class IntervalToTitle:
    def __init__(self):
        self.starting_points: List[int] = []
        self.titles: List[str] = []

    def append_interval(self, start: int, title: str) -> None:
        self.starting_points.append(start)
        self.titles.append(title)

    def get_title(self, index: int) -> str:
        if len(self.starting_points) == 0:
            raise IndexError("No intervals were set")

        if self.starting_points[0] > index:
            raise IndexError("Index is less then first starting point")

        for sp, title in zip(self.starting_points, self.titles):
            if index >= sp:
                return title

    def __str__(self) -> str:
        text = "{ "
        for i in range(len(self.starting_points) - 1):
            text += f"[{self.starting_points[i]}, {self.starting_points[i + 1]})"
            text += f" -> \"{self.titles[i]}\"\n  "
        text += f"[{self.starting_points[len(self.starting_points) - 1]}, ∞)"
        text += f" -> \"{self.titles[len(self.starting_points) - 1]}\"" + " }\n"
        return text


def main():
    model_name = 'roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)

    wiki_page = parse_wiki()
    i2t = IntervalToTitle()

    input_ids: List[int] = []

    for title, text in wiki_page.items():
        i2t.append_interval(len(input_ids), title)
        tokens = tokenizer.tokenize(text)
        input_ids += tokenizer.convert_tokens_to_ids(tokens)

    print(i2t)

    # Should we add <s> </s> tags?
    vector_len: int = 512
    padding_len: int = vector_len - (len(input_ids) % vector_len)
    input_ids += [tokenizer.pad_token_id] * padding_len  # add padding

    input_ids_tensor = torch.tensor(input_ids).reshape((-1, vector_len))
    output = model(input_ids_tensor)

    embeddings = output.last_hidden_state.detach()
    embeddings = embeddings.reshape((-1, embeddings.size()[2]))  # Squeeze batch dimension

    # We can cut padding before writing do disk / faiss

    df = pd.DataFrame(embeddings)
    df.to_csv('embeddings.csv')


if __name__ == '__main__':
    main()
