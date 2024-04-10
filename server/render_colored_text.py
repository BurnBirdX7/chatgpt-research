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
        color_num: int = 0
        last_chain: Optional[Chain] = None

        source_dict = defaultdict(list)
        token_list = []

        for i, key in enumerate(coloring.tokens):
            key: str

            if i in coloring.pos2chain:
                chain = coloring.pos2chain[i]
                source = chain.source
                score = chain.get_score()

                if last_chain != chain:
                    last_chain = chain
                    color_num += 1
                    source_dict[source].append(color_num)

                token_list.append({
                    "link": source,
                    "score": score,
                    "chain": str(chain),
                    "color_num": color_num,
                    "token": key
                })
            else:
                last_chain = None
                token_list.append({
                    "color_num": 0,
                    "token": key
                })

        source_list = list(sorted(source_dict.items(), key=lambda item: len(item[1]), reverse=True))

        result_list.append({
            "name": coloring.name,
            "sources": source_list,
            "token_coloring": token_list
        })
    return template_page.render(input_text=input_text, results=result_list)
