from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import Optional, List, Dict

from jinja2 import Template

from src.chaining import Chain


@dataclasses.dataclass
class Coloring:
    name: str
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
        accumulated = ""
        color_num: int = 1
        last_chain: Optional[Chain] = None

        def push_text():
            nonlocal color_num
            if last_chain is None:
                token_list.append({"color_num": 0, "token": accumulated})
            else:
                token_list.append(
                    {
                        "link": last_chain.source,
                        "score": last_chain.get_score(),
                        "chain": str(last_chain),
                        "color_num": color_num,
                        "token": accumulated,
                        "source_text": last_chain.matched_source_text,
                    }
                )

                color_num += 1
                source_dict[last_chain.source].append(color_num)

        for pos, token in enumerate(coloring.tokens):
            token: str

            chain = get_chain(pos)

            if chain == last_chain:
                accumulated += token
                continue

            push_text()

            last_chain = chain
            accumulated = token

        push_text()

        source_list = list(
            sorted(source_dict.items(), key=lambda item: len(item[1]), reverse=True)
        )

        result_list.append(
            {
                "name": coloring.name,
                "sources": source_list,
                "token_coloring": token_list,
            }
        )
    return template_page.render(input_text=input_text, results=result_list)
