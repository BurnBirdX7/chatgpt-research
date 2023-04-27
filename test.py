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
    target_page = wikipedia.page(title)
    print(len(target_page.text))
    d: Dict[str, str] = dict()

    for section in target_page.sections:
        d.update(traverse_sections(section, target_page.canonicalurl))

    return d


"""
Prints acquired data into embeddings.txt file
"""


class IntervalToSource:
    def __init__(self):
        self.starting_points: List[int] = []
        self.sources: List[str] = []

    def append_interval(self, start: int, source: str) -> None:
        self.starting_points.append(start)
        self.sources.append(source)

    def get_source(self, index: int) -> str:
        if len(self.starting_points) == 0:
            raise IndexError("No intervals were set")

        if self.starting_points[0] > index:
            raise IndexError("Index is less then first starting point")

        for sp, title in zip(self.starting_points, self.sources):
            if index >= sp:
                return title

    def __str__(self) -> str:
        text = "{ "
        for i in range(len(self.starting_points) - 1):
            text += f"[{self.starting_points[i]}, {self.starting_points[i + 1]})"
            text += f" -> \"{self.sources[i]}\"\n  "
        text += f"[{self.starting_points[len(self.starting_points) - 1]}, ∞)"
        text += f" -> \"{self.sources[len(self.starting_points) - 1]}\"" + " }\n"
        return text

    def to_csv(self, file: str):
        df = pd.DataFrame()
        df['Starting Point'] = self.starting_points
        df['Source'] = self.sources
        df.to_csv(file)


def main():
    model_name = 'roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)

    sections_dict = parse_wiki()
    i2t = IntervalToSource()
    input_ids: List[int] = []

    for title, text in sections_dict.items():
        i2t.append_interval(len(input_ids), title)
        tokens = tokenizer.tokenize(text)
        input_ids += tokenizer.convert_tokens_to_ids(tokens)

    i2t.to_csv('ranges.csv')

    # Should we add <s> </s> tags?
    vector_len: int = 512
    padding_len: int = vector_len - (len(input_ids) % vector_len)
    input_ids += [tokenizer.pad_token_id] * padding_len  # add padding

    input_ids_tensor = torch.tensor(input_ids).reshape((-1, vector_len))
    print("Computing text embedding...", end="")
    output = model(input_ids_tensor)
    print("Done")

    embeddings = output.last_hidden_state.detach()
    batch_size, sequence_length, embedding_len = embeddings.size()
    embeddings = embeddings.reshape((-1, embedding_len))  # Squeeze batch dimension

    # We can cut padding before writing do disk / faiss

    df = pd.DataFrame(embeddings)
    df.drop(df.index[-padding_len:])
    df.to_csv('embeddings.csv')


if __name__ == '__main__':
    main()
