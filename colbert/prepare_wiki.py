from __future__ import annotations

import dataclasses
import os
import sys
from typing import List, Dict
import xml.etree.ElementTree as ET
import mwparserfromhell as mw
from dataclasses import dataclass, field


@dataclass
class WikiContext:
    namespace: str | None = None
    current_title: str | None = None
    current_page: Dict[str, str] | None = None
    should_parse: bool = True
    passage_file = None
    source_file = None
    pid: int = 0
    flushed: int = 0

    def __flush_current_page(self) -> None:
        self.flushed += 1
        for heading, section in self.current_page.items():
            for paragraph in section.split('\n'):
                stripped_paragraph = paragraph.strip()
                if len(stripped_paragraph) > 0:
                    stripped_paragraph = stripped_paragraph.replace('\t', ' ')
                    self.passage_file.write(f"{self.pid}\t{stripped_paragraph}\n")
                    url = f"https://en.wikipedia.org/wiki/{self.current_title}#{heading.replace(' ', '_')}"
                    self.source_file.write(f"{self.pid}\t{url}\n")
                    self.pid += 1

    def new_page(self):
        if self.should_parse and self.current_page and self.current_title:
            self.__flush_current_page()

        self.current_title = None
        self.current_page = None
        self.should_parse = True

    def __tag(self, tag_name):
        return f"{{{self.namespace}}}{tag_name}" if self.namespace else tag_name

    @property
    def page_tag(self):
        return self.__tag("page")

    @property
    def title_tag(self):
        return self.__tag("title")

    @property
    def redirect_tag(self):
        return self.__tag("redirect")

    @property
    def text_tag(self):
        return self.__tag("text")

    @property
    def ns_tag(self):
        return self.__tag("ns")


def parse_wikitext(text: str) -> Dict[str, str]:
    """
    :param text: WikiText
    :return: Dictionary of sections, keys are section names and values are section texts
    """
    final_dict = {}
    wikicode = mw.parse(text, skip_style_tags=True)
    sections = wikicode.get_sections(flat=True)
    for section in sections:
        headings = section.filter_headings()
        text = section.strip_code(normalize=True)
        if len(headings) == 1:
            heading = headings[0].title.strip_code().strip()
            split_text = text.split('\n', 1)
            if len(split_text) == 1:
                continue
            text = split_text[1]  # Remove heading
        elif len(headings) == 0:
            heading = ""
        else:
            continue

        final_dict[heading] = str(text)

    return final_dict


def save_to_file(index_name: str, pages: Dict[str, Dict[str, str]]):
    with open(f"{index_name}_passages.tsv", "w") as passage_file, \
         open(f"{index_name}_sections.tsv", "w") as section_file:
        print("Opened files", passage_file, section_file)
        pid = 0

        print(f"Finished with pid = {pid}, cwd = {os.getcwd()}")


def prepare_wiki(args: List[str]):
    wiki_path = args.pop(0)

    print("Parsing Wikipedia")

    ctx = WikiContext()
    ctx.passage_file = open("test_passages.tsv", "w")
    ctx.source_file = open("test_sections.tsv", "w")

    for (event, elem) in ET.iterparse(wiki_path, events=("start", "start-ns", "end")):
        if event == "start-ns":
            if elem[0] == '':
                ctx.namespace = elem[1]
            continue

        if elem.tag == ctx.page_tag:
            if event == 'start':
                ctx.new_page()
                print(ctx.flushed)
        elif elem.tag == ctx.ns_tag:
            if event == 'end' and elem.text != '0':
                ctx.should_parse = False
        elif elem.tag == ctx.redirect_tag:
            ctx.should_parse = False
        elif elem.tag == ctx.title_tag:
            if event == 'end':
                ctx.current_title = elem.text
        elif elem.tag == ctx.text_tag:
            if event == 'end' and ctx.should_parse:
                ctx.current_page = parse_wikitext(elem.text)

        if ctx.flushed > 10000:
            break


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python -m colbert prepare_wiki [wiki_file]')
        exit(1)

    prepare_wiki(sys.argv[1:])
