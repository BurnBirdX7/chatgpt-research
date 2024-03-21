import copy
import math
import time
from datetime import datetime
from typing import Tuple, List, Set, Optional, Dict

import numpy as np  # type: ignore
import faiss  # type: ignore
import wikipediaapi  # type: ignore
import torch
from jinja2 import Template

from scripts._elvis_data import elvis_related_articles
from src import Roberta, EmbeddingsBuilder, Index, OnlineWiki
from transformers import RobertaForMaskedLM

from src.config import EmbeddingBuilderConfig, WikiConfig, IndexConfig
from src.token_chain import Chain

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


def generate_sequences(source_len: int, likelihoods: torch.Tensor,
                       token_ids: List[int], token_start_pos: int, source: str) -> List[Chain]:
    """
    Generates chains of tokens with the same source
    """
    result_chains: List[Chain] = []

    for source_start_pos in range(0, source_len):
        chain = Chain(source)
        shift_upper_bound = min(source_len - source_start_pos, len(token_ids) - token_start_pos)
        for shift in range(0, shift_upper_bound):
            token_pos = token_start_pos + shift
            source_pos = source_start_pos + shift

            assert token_pos < len(token_ids)
            assert source_pos < source_len

            token_curr_id = token_ids[token_pos]
            token_curr_likelihood = likelihoods[source_pos][token_curr_id].item()

            if token_curr_likelihood < 1e-5:
                chain.skip()
                if chain.skips > 3:
                    break
            else:
                chain.append(token_curr_likelihood, token_pos)
                if len(chain) > 1:
                    result_chains.append(copy.deepcopy(chain))

    return result_chains


def color_main_with_chaining(input_text: str,
                             wiki_config: WikiConfig,
                             index: Index,
                             embeddings_config: EmbeddingBuilderConfig) -> str:
    """
    Colors text when probable sources already defined
    :param input_text: text to color
    :param wiki_config: wikipedia articles that could be sources for the text
    :param index: index that contains all known sources
    :param embeddings_config: config RoBERTa embeddings, `normalize`'s value is ignored
    """
    embeddings_config.normalize = True
    embeddings_builder = EmbeddingsBuilder(embeddings_config)
    input_embeddings = embeddings_builder.from_text(input_text)
    print(input_embeddings)
    faiss.normalize_L2(input_embeddings)  # TODO: Move normalization to the builder

    sources, result_dists = index.get_embeddings_source(input_embeddings)
    print("sources:", sources, "result_ids dists:", result_dists, "\n\n")

    tokenized_input = tokenizer.tokenize(input_text)  # разбиваем на токены входную строку с гпт
    print("tokens:", tokenized_input, "\n\n")  # все токены разбитые из input

    gpt_token_ids = tokenizer.convert_tokens_to_ids(tokenized_input)
    sourcetext_dict = OnlineWiki.get_sections(wiki_config.target_pages)

    start_time = time.perf_counter()
    result_chains = []
    for token_pos, (token, token_id, source) in enumerate(zip(tokenized_input, gpt_token_ids, sources)):
        wiki_text = sourcetext_dict[source]
        wiki_token_ids = tokenizer.encode(wiki_text, return_tensors='pt').squeeze()

        print(f"> token: '{token}', id: {token_id}, source token count: {len(wiki_token_ids)}, top source: {source}, ")
        for batch in range(0, len(wiki_token_ids), 256):
            print(f"\tbatch: [{batch} : {batch + 512})")
            wiki_token_ids_batch = wiki_token_ids[batch:batch + 512]
            if len(wiki_token_ids_batch) < 2:
                break

            wiki_token_ids_batch = wiki_token_ids_batch.unsqueeze(0)

            with torch.no_grad():
                output_page = modelMLM(wiki_token_ids_batch)

            wiki_logits = output_page[0].squeeze()
            likelihoods = torch.nn.functional.softmax(wiki_logits, dim=1)
            result_chains += generate_sequences(len(wiki_logits), likelihoods, gpt_token_ids, token_pos, source)

    print("All sequences: ")
    for chain in result_chains:
        print(chain)

    filtered_chains: List[Chain] = []
    marked_positions: Set[int] = set()
    for chain in sorted(result_chains, key=lambda x: x.get_score(), reverse=True):
        positions = chain.get_token_positions()
        marked_positions_inside_chain = marked_positions.intersection(positions)
        if len(marked_positions_inside_chain) == 0:
            marked_positions |= positions
            filtered_chains.append(chain)

    print("Filtered chains:")
    for chain in filtered_chains:
        print(chain)

    print(f"Time: {time.perf_counter() - start_time} s.")

    # prepare tokens for coloring
    tokens_for_coloring = map(lambda s: tokenizer.convert_tokens_to_string([s]), tokenized_input)

    # prepare links for coloring
    pos2chain: Dict[int, Chain] = {}
    for i, chain in enumerate(filtered_chains):
        for pos in chain.get_token_positions():
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
    result_html = template_page.render(result=output_page, gpt_response=input_text, list_of_colors=output_source_list)

    file = datetime.now().strftime("./result-%Y%m%d-%H%M%S.html")

    with open(file, "w", encoding="utf-8") as f:
        f.write(result_html)

    return file


if __name__ == "__main__":
    # GPT-3.5 Output:
    text = (
        "Presley's father Vernon was of German, Scottish, and English origins, and a descendant of the Harrison family "
        "of Virginia through his mother, Minnie Mae Presley (née Hood). Presley's mother Gladys was Scots-Irish with "
        "some French Norman ancestry. She and the rest of the family believed that her great-great-grandmother,"
        " Morning Dove White, was Cherokee. This belief was restated by Elvis's granddaughter Riley Keough in 2017. "
        "Elaine Dundy, in her biography, supports the belief."
    )

    index = Index.load(IndexConfig())  # Use default config, TODO: Maybe change
    embeddings_builder = EmbeddingsBuilder(EmbeddingBuilderConfig())
    color_main_with_chaining(text, WikiConfig(elvis_related_articles))
