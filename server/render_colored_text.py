import dataclasses
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

    color_num: int = 7
    last_chain: Optional[Chain] = None

    result_list = []

    for coloring in colorings:

        source_list = []
        token_list = []

        result_list.append({
            "sources": source_list,
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
                    source_list.append({
                        "link": source,
                        "color": f"color{color_num}"
                    })
                    color_num += 1

                token_list.append({
                    "link": source,
                    "score": score,
                    "chain": str(chain),
                    "color": f"color{color_num}",
                    "token": key
                })
            else:
                last_chain = None
                token_list.append({
                    "color": f"color{color_num}",
                    "token": key
                })

    return template_page.render(input_text=input_text, results=result_list)
