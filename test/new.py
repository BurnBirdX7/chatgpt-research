import copy
import math
import time
from typing import Tuple, List, Set, Optional

import numpy as np  # type: ignore
import faiss  # type: ignore
import wikipediaapi  # type: ignore
import torch
from jinja2 import Template

from src import Roberta, Config, SourceMapping, Embeddings, Index, Wiki
from transformers import RobertaTokenizer, RobertaForMaskedLM

tokenizer, model = Roberta.get_default()
modelMLM = RobertaForMaskedLM.from_pretrained('roberta-large')
batched_token_ids = torch.empty((1, 512), dtype=torch.int)

page_template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Result</title>
    <link rel="stylesheet" type="text/css" href="../static/style_result.css">
</head>
<body>
<h1>Result of research</h1>
<pre><b>Input text:</b></pre>
{{ gpt_response }}
<pre><b>Top paragraphs:</b></pre>
{{ list_of_colors }}
<pre><b>Result:</b></pre>
{{ result }}
</body>
</html>
"""

source_link_template_str = "<a href=\"{{ link }}\" class=\"{{ color }}\">{{ token }}</a>"
source_text_template_str = "<a class=\"{{ color }}\"><i>{{ token }}</i></a>"
source_item_str = "<a href=\"{{ link }}\" class=\"{{ color }}\">{{ token }}</a></br>"


def get_page_section_from_wiki(source: str) -> str:
    wikipedia = wikipediaapi.Wikipedia("en")
    flag = False
    if "#" not in source:
        flag = True
        title = source.split("https://en.wikipedia.org/wiki/")[1].replace("_", " ")
    else:
        title = source.split("https://en.wikipedia.org/wiki/")[1].split("#")[0].replace("_", " ")

    target_page = wikipedia.page(title)
    if flag:
        section = target_page.summary
    else:
        section = target_page.section_by_title \
            (source.split("https://en.wikipedia.org/wiki/")[1].split("#")[1].replace("_",
                                                                                     " "))  # возврощает последнюю секцию

    return str(section)


class Chain:
    string: str
    likelihoods: List[float]
    positions: List[int]
    source: str

    def __init__(self, string: str, likelihoods: List[float], positions: List[int], source: str):
        assert (len(likelihoods) == len(positions))
        self.string = string
        self.likelihoods = likelihoods
        self.positions = positions
        self.source = source

    def __len__(self) -> int:
        return len(self.positions)

    def extend(self, string: str, likelihood: float, position: int) -> "Chain":
        return Chain(self.string + string,
                     self.likelihoods + [likelihood],
                     self.positions + [position],
                     self.source)

    def get_score(self):
        l = len(self)

        # log2(2 + len) * ((lik_h_0 * ... * lik_h_len) ^ 1 / len) - score
        score = 1.0
        for lh in self.likelihoods:
            score *= lh

        score **= 1 / l
        score *= math.log2(2 + l)
        return score


# GLOBAL result sequence:
result_sequence: List[Chain] = []


def generate_sequences(chain: Chain, last_hidden_state: torch.Tensor, start_idx: int, tokens: List[str], token_pos):

    if start_idx >= len(last_hidden_state) or token_pos >= len(tokens):
        if len(chain) > 1:
            result_sequence.append(chain)

        return

    for idx in range(start_idx, len(last_hidden_state)):
        probs = torch.nn.functional.softmax(last_hidden_state[idx])
        column_tokens = torch.topk(last_hidden_state[idx], k=20, dim=0)
        top_twenty_tokens = column_tokens[1]
        probability_top_twenty_tokens = [probs[i] for i in top_twenty_tokens]
        words = [tokenizer.decode(i.item()).strip() for i in top_twenty_tokens]

        token = tokens[token_pos]
        if "Ġ" in token:
            token = token.split("Ġ")[1]

        for word_idx in range(len(words)):
            if words[word_idx] == token:
                prob = probability_top_twenty_tokens[word_idx].item()
                current_chain = chain.extend(token, prob, token_pos)
                if prob >= 0.1:
                    generate_sequences(current_chain, last_hidden_state, idx + 1,
                                       tokens, token_pos + 1)
                else:
                    if len(current_chain) > 1:
                        result_sequence.append(current_chain)


def cast_output(tokens, source_link):
    for key, src in enumerate(tokens):
        print(key, src)
        print(source_link[key])

    for i, key, src in enumerate(zip(tokens, source_link)):
        print("::", i, "::", key, "::", src)

    template = Template(source_link_template_str)
    output = ''
    for i, key in enumerate(tokens):
        value_from_map1 = tokens[key]
        value_from_map2 = source_link[key]
        print(value_from_map1, value_from_map2)
        # Check if the key is present in the second map


def main(gpt_response) -> None:
    index = Index.load(Config.index_file, Config.mapping_file)

    embeddings = Embeddings(tokenizer, model).from_text(gpt_response)
    print(embeddings)
    faiss.normalize_L2(embeddings)

    result_dists, result_ids = index.index.search(embeddings, 1)
    print("indexes:", result_ids, "result_ids dists:", result_dists, "\n\n")

    tokens = tokenizer.tokenize(gpt_response)  # разбиваем на токены входную строку с гпт
    print("tokens:", tokens, "\n\n")  # все токены разбитые из input

    start = time.perf_counter()
    for token_pos, token in enumerate(tokens):
        source = index.get_source(int(result_ids[token_pos]))
        print("source: ", source)
        text_from_section = get_page_section_from_wiki(source)  # последняя секция из вики с документа

        if "Ġ" in token:
            token = token.split("Ġ")[1]
        print("token:", token)

        token_ids = tokenizer.encode(text_from_section, return_tensors='pt')

        for batch in range(0, token_ids.shape[1], 511):
            batched_token_ids = token_ids[0, batch:batch + 512].unsqueeze(0)

            with torch.no_grad():
                output_page = modelMLM(batched_token_ids)

            last_hidden_state = output_page[0].squeeze()

            empty_chain = Chain("", [], [], source)
            generate_sequences(empty_chain, last_hidden_state, 0, tokens, token_pos)

            print("probe res::", result_sequence)

    print("whole res::", result_sequence)
    print("shape_of_end_sequence:::", len(result_sequence))

    sorted_result_chain = sorted(result_sequence, key=lambda x: x.get_score(), reverse=True)
    print("sorting:::", sorted_result_chain)

    filtered_chains: List[Chain] = []
    marked_positions: Set[int] = set()
    for chain in sorted_result_chain:
        marked_in_chain = marked_positions.intersection(chain.positions)
        if len(marked_in_chain) == 0:
            marked_positions |= set(chain.positions)
            filtered_chains.append(chain)

    print("whole sequence:", filtered_chains)
    print("whole time:", time.perf_counter() - start)

    # prepare tokens for coloring
    tokens_for_coloring = map(lambda s: s.replace('Ġ', ' ').replace('Ċ', '</br>'), tokens)

    # prepare links for coloring
    pos2source = {}
    for i, chain in enumerate(filtered_chains):
        for pos in chain.positions:
            print(f"pos {pos}, source {chain.source}")
            pos2source[pos] = chain.source

    template_page = Template(page_template_str)
    template_link = Template(source_link_template_str)
    template_text = Template(source_text_template_str)
    template_source_item = Template(source_item_str)

    color = 7
    output_page = ''
    output_source_list = ''
    last_source: Optional[str] = None
    for i, key in enumerate(tokens_for_coloring):
        key: str

        if i in pos2source:
            if last_source == pos2source[i]:
                output_page += template_link.render(link=pos2source[i],
                                                    color="color" + str(color),
                                                    token=key[1:-1])
            else:
                # New sequence:
                color += 1
                output_source_list += template_source_item.render(link=pos2source[i],
                                                                  color="color" + str(color),
                                                                  token=pos2source[i])
                last_source = pos2source[i]
                output_page += template_link.render(link=pos2source[i],
                                                    color="color" + str(color),
                                                    token=key[1:-1])
        else:
            last_source = None
            output_page += template_text.render(token=key[1:-1], color="color0")

    output_source_list += '</br>'
    result_html = template_page.render(result=output_page, gpt_response=gpt_response, list_of_colors=output_source_list)

    with open("./server/templates/template_of_result_page.html", "w", encoding="utf-8") as f:
        f.write(result_html)


if __name__ == "__main__":
    main(
        "Presley's father Vernon was of German, Scottish, and English origins, and a descendant of the Harrison family "
        "of Virginia through his mother, Minnie Mae Presley (née Hood). Presley's mother Gladys was Scots-Irish with "
        "some French Norman ancestry. She and the rest of the family believed that her great-great-grandmother,"
        " Morning Dove White, was Cherokee. This belief was restated by Elvis's granddaughter Riley Keough in 2017. "
        "Elaine Dundy, in her biography, supports the belief.")  # gpt output
