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
        color_num: int = 1
        last_chain: Optional[Chain] = None

        source_dict = defaultdict(lambda: 0)
        token_list = []
        result_list.append({
            "sources": source_dict,
            "token_coloring": token_list
        })

        for i, key in enumerate(coloring.tokens):
            key: str

            if i in coloring.pos2chain:
                chain = coloring.pos2chain[i]
                source = chain.source
                score = chain.get_score()

                if last_chain != chain:
                    last_chain = chain
                    source_dict[source] += 1

                    if color_num == 1:
                        color_num = 2
                    else:
                        color_num = 1

                token_list.append({
                    "link": source,
                    "score": score,
                    "chain": str(chain),
                    "color": f"type_{color_num}",
                    "token": key
                })
            else:
                last_chain = None
                token_list.append({
                    "color": f"type_0",
                    "token": key
                })

    return template_page.render(input_text=input_text, results=result_list)
