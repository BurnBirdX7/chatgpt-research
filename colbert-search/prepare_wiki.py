from __future__ import annotations

import dataclasses
import os
import sys
import typing
from typing import List, Dict, TextIO
import xml.etree.ElementTree as ET
import mwparserfromhell as mw
from dataclasses import dataclass


banned_title_prefixes: list[str] = [
    "Category:", "File:", "See also", "References", "External links"
]

def is_banned(title: str) -> bool:
    for banned in banned_title_prefixes:
        if title.strip().startswith(banned):
            return True

    return False


@dataclass
class WikiParseContext:
    collection_name: str = "default"
    namespace: str | None = None
    current_title: str | None = None
    current_page: Dict[str, str] | None = None
    should_parse: bool = True
    passage_file: TextIO = None
    source_file: TextIO = None
    pid: int = 0
    flushed: int = 0

    def __flush_current_page(self) -> None:
        if not self.should_parse or self.current_page is None or self.current_title is None:
            return

        self.flushed += 1
        for heading, section in self.current_page.items():
            for paragraph in section.split('\n'):
                stripped_paragraph = paragraph.strip()
                if len(stripped_paragraph) > 0:
                    stripped_paragraph = stripped_paragraph.replace('\t', ' ')  # make tsv-safe
                    url = f"https://en.wikipedia.org/wiki/{self.current_title}#{heading.replace(' ', '_')}"
                    self.passage_file.write(f"{self.pid}\t{stripped_paragraph}\n")
                    self.source_file.write(f"{self.pid}\t{url}\n")
                    self.pid += 1

        self.current_title = None
        self.current_page = None

    def new_page(self):
        self.__flush_current_page()
        self.current_title = None
        self.current_page = None
        self.should_parse = True

    def new_files(self, num: int) -> None:
        self.close_files()
        self.source_file = open(f"{self.collection_name}_sources_{num}.tsv", "w")
        self.passage_file = open(f"{self.collection_name}_passages_{num}.tsv", "w")
        self.pid = 0
        self.flushed = 0

    def close_files(self) -> None:
        self.__flush_current_page()
        if self.source_file is not None:
            self.source_file.close()
        if self.passage_file is not None:
            self.passage_file.close()

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
    :param text: WikiText, encoding=""
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


def report_page(n: int) -> None:
    print('.', end='')
    if n % 100 == 0:
        print(n)

def prepare_wiki(args: List[str]):
    collection_name = args[0]
    wiki_path = args[1]

    print("Parsing Wikipedia")

    file_num = 1

    ctx = WikiParseContext()
    ctx.collection_name = collection_name
    ctx.new_files(file_num)

    for (event, elem) in ET.iterparse(wiki_path, events=("start", "start-ns", "end")):
        if event == "start-ns":
            print("new namespace", elem)
            if elem[0] == '':
                print(f"set namespace from {ctx.namespace} to {elem[1]}")
                ctx.namespace = elem[1]
            continue

        if elem.tag == ctx.page_tag:
            if event == 'start':
                ctx.new_page()
                report_page(ctx.flushed)
        elif elem.tag == ctx.ns_tag:
            if event == 'end' and elem.text != '0':
                ctx.should_parse = False
        elif elem.tag == ctx.redirect_tag:
            ctx.should_parse = False
        elif elem.tag == ctx.title_tag:
            if event == 'end':
                ctx.current_title = elem.text
        elif elem.tag == ctx.text_tag:
            if event == 'end' and ctx.should_parse and not is_banned(ctx.current_title):
                ctx.current_page = parse_wikitext(elem.text)

        if ctx.flushed > 10000:
            print(f"==== flushed {ctx.flushed} passages, creating new file ====", flush=True)
            file_num += 1
            ctx.new_files(file_num)

        ctx.close_files()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python -m colbert-search prepare_wiki [collection_name] [wiki_file]')
        exit(1)

    prepare_wiki(sys.argv[1:])
