import copy
import math
import time
from typing import Tuple, List, Set, Optional, Dict

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

source_link_template_str = "<a href=\"{{ link }}\" class=\"{{ color }}\" title=\"score: {{score}}\">{{ token }}</a>\n"
source_text_template_str = "<a class=\"{{ color }}\"><i>{{ token }}</i></a>\n"
source_item_str = "<a href=\"{{ link }}\" class=\"{{ color }}\">{{ link }}</a></br>\n"


class Chain:
    likelihoods: List[float]
    positions: List[int]
    source: str

    def __init__(self, likelihoods: List[float], positions: List[int], source: str):
        assert (len(likelihoods) == len(positions))
        self.likelihoods = likelihoods
        self.positions = positions
        self.source = source

    def __len__(self) -> int:
        return len(self.positions)

    def __str__(self) -> str:
        return (f"Chain {{ pos: {self.positions}, likelihoods: {self.likelihoods}, "
                f"score: {self.get_score()}, source: {self.source} }}")

    def __repr__(self) -> str:
        return str(self)

    def extend(self, likelihood: float, position: int) -> "Chain":
        return Chain(self.likelihoods + [likelihood],
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
result_chains: List[Chain] = []


def generate_sequences(chain: Chain, last_hidden_state: torch.Tensor, probs: torch.Tensor, top20: torch.Tensor,
                       start_idx: int, tokens: List[int], token_pos: int):

    if start_idx >= len(last_hidden_state) or token_pos >= len(tokens):
        if len(chain) > 1:
            result_chains.append(chain)
        return

    for idx in range(start_idx, len(last_hidden_state)):
        token_curr = tokens[token_pos]

        for token_id_t20 in top20[idx]:
            if token_id_t20.item() != token_curr:
                continue

            prob = probs[idx][token_id_t20].item()
            current_chain = chain.extend(prob, token_pos)
            if prob >= 0.1:
                generate_sequences(current_chain, last_hidden_state, probs, top20,
                                   idx + 1, tokens, token_pos + 1)
            else:
                if len(current_chain) > 1:
                    result_chains.append(current_chain)


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

    sources, result_dists = index.get_embeddings_source(embeddings)
    print("soureces:", sources, "result_ids dists:", result_dists, "\n\n")

    gpt_tokens = tokenizer.tokenize(gpt_response)  # разбиваем на токены входную строку с гпт
    print("tokens:", gpt_tokens, "\n\n")  # все токены разбитые из input

    gpt_token_ids = tokenizer.convert_tokens_to_ids(gpt_tokens)

    wiki_dict = dict()
    for page in Config.page_names:
        wiki_dict |= Wiki.parse(page)

    start = time.perf_counter()
    for token_pos, (token, token_id, source) in enumerate(zip(gpt_tokens, gpt_token_ids, sources)):
        wiki_text = wiki_dict[source]
        wiki_token_ids = tokenizer.encode(wiki_text, return_tensors='pt').squeeze()

        print(f"> token: {token}, id: {token_id}, top source: {source}")
        for batch in range(0, len(wiki_token_ids), 511):
            print(f"\tbatch: [{batch} : {batch + 512})")
            wiki_token_ids_batch = wiki_token_ids[batch:batch + 512].unsqueeze(0)

            with torch.no_grad():
                output_page = modelMLM(wiki_token_ids_batch)

            last_hidden_state = output_page[0].squeeze()
            probs = torch.nn.functional.softmax(last_hidden_state, dim=1)
            top20 = torch.topk(last_hidden_state, k=20, dim=1).indices

            empty_chain = Chain([], [], source)
            generate_sequences(empty_chain, last_hidden_state, probs, top20, 0, gpt_token_ids, token_pos)

    print("All sequences: ")
    for chain in result_chains:
        print(chain)

    filtered_chains: List[Chain] = []
    marked_positions: Set[int] = set()
    for chain in sorted(result_chains, key=lambda x: x.get_score(), reverse=True):
        marked_in_chain = marked_positions.intersection(chain.positions)
        if len(marked_in_chain) == 0:
            marked_positions |= set(chain.positions)
            filtered_chains.append(chain)

    print("Filtered chains:")
    for chain in filtered_chains:
        print(chain)

    print(f"Time: {time.perf_counter() - start} s.")

    # prepare tokens for coloring
    tokens_for_coloring = map(lambda s: s.replace('Ġ', ' ').replace('Ċ', '</br>'), gpt_tokens)

    # prepare links for coloring
    pos2chain: Dict[int, Chain] = {}
    for i, chain in enumerate(filtered_chains):
        for pos in chain.positions:
            pos2chain[pos] = chain

    template_page = Template(page_template_str)
    template_link = Template(source_link_template_str)
    template_text = Template(source_text_template_str)
    template_source_item = Template(source_item_str)

    color: int = 7
    output_page: str = ''
    output_source_list: str = ''
    last_chain: Optional[Chain] = None
    for i, key in enumerate(tokens_for_coloring):
        key: str

        if i in pos2chain:
            chain = pos2chain[i]
            source = chain.source
            score = chain.get_score()
            if last_chain == chain:
                output_page += template_link.render(link=source,
                                                    score=score,
                                                    color="color" + str(color),
                                                    token=key)
            else:
                color += 1
                last_chain = chain
                output_source_list += template_source_item.render(link=source,
                                                                  color="color" + str(color))
                output_page += template_link.render(link=source,
                                                    score=score,
                                                    color="color" + str(color),
                                                    token=key)
        else:
            last_chain = None
            output_page += template_text.render(token=key, color="color0")

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
