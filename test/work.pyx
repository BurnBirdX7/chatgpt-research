import torch
import time
import math
import numpy as np

from src import Roberta, Config, SourceMapping, Embeddings, Index, Wiki
from transformers import RobertaTokenizer, RobertaForMaskedLM
import faiss  # type: ignore
import wikipediaapi  # type: ignore
import math

cdef tokenizer, model = Roberta.get_default()
cdef modelMLM = RobertaForMaskedLM.from_pretrained('roberta-large')
cdef batched_token_ids = torch.empty((1, 512), dtype=torch.int)

cdef str page_template_str = """
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

cdef str source_link_template_str = "<a href=\"{{ link }}\" class=\"{{ color }}\" title=\"score: {{score}}\">{{ token }}</a>\n"
cdef str source_text_template_str = "<a class=\"{{ color }}\"><i>{{ token }}</i></a>\n"
cdef str source_item_str = "<a href=\"{{ link }}\" class=\"{{ color }}\">{{ link }}</a></br>\n"

# GLOBAL result sequence:
cdef result_chains = []

cdef class Chain:
    cdef public list likelihoods
    cdef public list[] positions
    cdef public str source

    cpdef init(self, likelihoods: [float], positions: [int], source: str):
        assert (len(likelihoods) == len(positions))
        self.likelihoods = likelihoods
        self.positions = positions
        self.source = source

    cpdef int len(self):
        return len(self.positions)

    def __len__(self) -> int:
        return len(self.positions)

    cpdef get_score(self):
        cdef double score
        cdef int l = self.len()

        # log2(2 + len) * ((lik_h_0 * ... * lik_h_len) ^ 1 / len) - score
        score = 1.0
        for lh in self.likelihoods:
            score *= lh

        score **= 1 / l
        score *= math.log2(2 + l)
        return score


    cpdef extend(self, likelihood, position):
        return Chain(self.likelihoods + [likelihood],
                     self.positions + [position],
                     self.source)


cpdef generate_sequences(chain, last_hidden_state, probs,
                       start_idx, tokens, token_pos):
    cdef int idx
    cdef int token_curr
    cdef double prob
    cdef Chain current_chain
    if start_idx >= len(last_hidden_state) or token_pos >= len(tokens):
        if len(chain) > 1:
            result_chains.append(chain)
        return

    for idx in range(start_idx, len(last_hidden_state)):
        token_curr = tokens[token_pos]
        prob = probs[idx][token_curr].item()
        if prob >= 0.05:
            current_chain = chain.extend(prob, token_pos)
            generate_sequences(current_chain, last_hidden_state, probs, idx + 1, tokens, token_pos + 1)
        else:
            if len(chain.positions) > 1:
                result_chains.append(chain)

from cython.parallel cimport prange

cpdef res(gpt_response: str):
    cdef   wiki_text
    cdef   wiki_token_ids_batch
    cdef   last_hidden_state
    cdef index
    cdef embeddings
    print("start")
    index = Index.load(Config.index_file, Config.mapping_file)
    embeddings = Embeddings(tokenizer, model).from_text(gpt_response)
    print(embeddings)
    faiss.normalize_L2(embeddings)
    sources, result_dists = index.get_embeddings_source(embeddings)
    print("soureces:", sources, "result_ids dists:", result_dists, "\n\n")

    gpt_tokens = tokenizer.tokenize(gpt_response)
    print("tokens:", gpt_tokens, "\n\n")

    gpt_token_ids = tokenizer.convert_tokens_to_ids(gpt_tokens)

    wiki_dict = dict()
    for page in Config.page_names:
        wiki_dict |= Wiki.parse(page)


    for token_pos, (token, token_id, source) in enumerate(zip(gpt_tokens, gpt_token_ids, sources)):

        wiki_text = wiki_dict[source]
        wiki_token_ids = tokenizer.encode(wiki_text, return_tensors='pt').squeeze()

        for batch in range(0, len(wiki_token_ids), 511):
            wiki_token_ids_batch = wiki_token_ids[batch:batch + 512].unsqueeze(0)

            with torch.no_grad():
                output_page = modelMLM(wiki_token_ids_batch)

            last_hidden_state = output_page[0].squeeze()
            probs = torch.nn.functional.softmax(last_hidden_state, dim=1)

            empty_chain = Chain([], [], source)
            generate_sequences(empty_chain, last_hidden_state, probs, 0, gpt_token_ids, token_pos)

    print("All sequences: ")
    for chain in result_chains:
        print(chain)
    return result_chains
