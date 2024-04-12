from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import Optional, List, Dict

from jinja2 import Template

from src.chaining import Chain


@dataclasses.dataclass
class Coloring:
    title: str
    pipeline_name: str
    tokens: List[str]
    pos2chain: Dict[int, Chain]


def render_colored_text(input_text: str, colorings: List[Coloring]) -> str:
    template_page = Template(open("server/templates/result_page.html.j2", "r").read())

    result_list = []

    for coloring in colorings:

        def get_chain(pos: int) -> Chain | None:
            if pos not in coloring.pos2chain:
                return None
            return coloring.pos2chain[pos]

        source_dict = defaultdict(list)
        token_list = []

        # Track progress
        color_num: int = 0
        last_chain: Optional[Chain] = None
        relative_pos = 0

        for pos, token in enumerate(coloring.tokens):
            chain = get_chain(pos)

            if chain is None:
                token_list.append({"color_num": 0, "token": token})
                continue

            if chain is not last_chain:
                color_num += 1
                source_dict[chain.source].append(color_num)
                last_chain = chain
                relative_pos = 0

            token_list.append(
                {
                    "url": chain.source,
                    "color_num": color_num,
                    "score": chain.get_score(),
                    "target_pos": pos,
                    "target_likelihood": chain.get_target_likelihood(pos),
                    "target_text": chain.attachment["target_text"],
                    "source_text": chain.attachment["source_text"],
                    "chain": str(chain),
                    "token": token,
                    "source_token": chain.attachment["source_tokens"][relative_pos]
                }
            )
            relative_pos += 1

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
