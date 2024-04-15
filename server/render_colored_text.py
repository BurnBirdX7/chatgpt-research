from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import List

from jinja2 import Template

from src.chaining import Chain


@dataclasses.dataclass
class Coloring:
    title: str
    pipeline_name: str
    tokens: List[str]
    chains: List[Chain]


def render_colored_text(input_text: str, colorings: List[Coloring]) -> str:
    template_page = Template(open("server/templates/result_page.html.j2", "r").read())

    result_list = []

    for coloring in colorings:

        sorted_chains = sorted(coloring.chains, key=lambda ch: ch.target_begin_pos)

        source_dict = defaultdict(list)
        token_list = []

        # Track progress
        last_target_pos = 0

        for color_num, chain in enumerate(sorted_chains, 1):
            source_tokens = chain.attachment["source_tokens"]
            source_dict[chain.source].append(color_num)

            for t_pos in range(last_target_pos, chain.target_begin_pos):
                token_list.append({"color_num": 0, "token": coloring.tokens[t_pos], "target_pos": t_pos})

            source_matches = chain.source_matches()
            for t_pos in range(chain.target_begin_pos, chain.target_end_pos):
                s_pos = source_matches[t_pos]
                if s_pos is None:
                    matched_token = "[skipped]"
                else:
                    matched_token = source_tokens[s_pos]
                token_list.append(
                    {
                        "url": chain.source,
                        "color_num": color_num,
                        "score": chain.get_score(),
                        "target_pos": t_pos,
                        "target_likelihood": chain.get_target_likelihood(t_pos),
                        "target_text": chain.attachment["target_text"],
                        "source_text": chain.attachment["source_text"],
                        "source_pos": s_pos,
                        "chain": str(chain),
                        "token": coloring.tokens[t_pos],
                        "source_token": matched_token,
                    }
                )

            last_target_pos = chain.target_end_pos

        source_list = list(sorted(source_dict.items(), key=lambda item: len(item[1]), reverse=True))

        result_list.append(
            {
                "name": coloring.title,
                "key": coloring.pipeline_name,
                "sources": source_list,
                "token_coloring": token_list,
            }
        )
    return template_page.render(input_text=input_text, results=result_list)
