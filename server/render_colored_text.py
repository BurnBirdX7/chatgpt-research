from typing import Optional, List, Dict

from jinja2 import Template

from src.token_chain import Chain


def render_colored_text(input_text: str, tokens: List[str], pos2chain: Dict[int, Chain]) -> str:
    template_page = Template(open("templates/result_page.jinja2.html", "r").read())
    template_link = Template(open("templates/source_link.jinja2.html", "r").read())
    template_text = Template(open("templates/source_text.jinja2.html", "r").read())
    template_source_item = Template(open("templates/source_item.jinja2.html.jinja2.html", "r").read())

    color: int = 7
    output_page: str = ''
    output_source_list: str = ''
    last_chain: Optional[Chain] = None
    for i, key in enumerate(tokens):
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
    return template_page.render(result=output_page, input_text=input_text, list_of_colors=output_source_list)
