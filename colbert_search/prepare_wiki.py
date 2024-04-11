from __future__ import annotations

import os.path
import sys
from typing import Dict, TextIO, List
import xml.etree.ElementTree as ET
import mwparserfromhell as mw
from dataclasses import dataclass

from src.pipeline import BaseNode, ListDescriptor
from src import WikiDataFile, SourceMapping

banned_title_prefixes: list[str] = [
    "Category:",
    "File:",
    "See also",
    "References",
    "External links",
]


def is_title_banned(title: str) -> bool:
    for banned in banned_title_prefixes:
        if title.strip().startswith(banned):
            return True

    return False


@dataclass
class WikiParseContext:
    # General:
    collection_name: str = "default"
    output_dir: str = os.path.abspath("./")
    namespace: str | None = None

    # Current "session"
    passage_file: TextIO = None
    passage_file_path: str = None
    source_mapping: SourceMapping = None
    source_mapping_path: str = None
    pid: int = 0
    pushed: int = 0

    # Current source:
    current_title: str | None = None
    current_page: Dict[str, str] | None = None
    should_parse: bool = True

    def push(self) -> None:
        """
        Pushes data from text representation of wikipedia article into file and mapping structures
        """
        if (
            not self.should_parse
            or self.current_page is None
            or self.current_title is None
        ):
            return

        self.pushed += 1
        for heading, section in self.current_page.items():
            if is_title_banned(heading):
                continue

            start_pid = self.pid
            for paragraph in section.split("\n"):
                stripped_paragraph = paragraph.strip()
                if len(stripped_paragraph) > 0:
                    stripped_paragraph = stripped_paragraph.replace(
                        "\t", " "
                    )  # make tsv-safe
                    self.passage_file.write(f"{self.pid}\t{stripped_paragraph}\n")
                    self.pid += 1

            clean_title = self.current_title.replace(" ", "_")
            clean_header = heading.replace(" ", "_").replace("\t", "_").replace("'", "")
            if "\n" in clean_header:
                # Header is malformed, remove part of it
                clean_header = clean_header.split("\n", maxsplit=1)[0]
            url = f"https://en.wikipedia.org/wiki/{clean_title}#{clean_header}"
            self.source_mapping.append_interval(self.pid - start_pid, url)

        self.current_title = None
        self.current_page = None

    def new_page(self):
        self.push()
        self.current_title = None
        self.current_page = None
        self.should_parse = True

    def reset(self, num: int) -> str:
        """
        Creates new files to output passages and source mapping for the passages

        :param num: arbitrary number to distinguish between iterations
        :return: name of the new file
        """
        self.flush()
        if self.passage_file is not None:
            self.passage_file.close()

        self.source_mapping_path = os.path.join(
            self.output_dir, f"{self.collection_name}_sources_{num}.tsv"
        )
        self.source_mapping = SourceMapping()

        self.passage_file_path = os.path.join(
            self.output_dir, f"{self.collection_name}_passages_{num}.csv"
        )
        self.passage_file = open(self.passage_file_path, "w")

        self.pid = 0
        self.pushed = 0

        return self.passage_file.name

    def flush(self):
        """
        Flushes accumulated data to the disk
        """
        self.push()
        if self.passage_file is not None:
            self.passage_file.flush()
        if self.source_mapping is not None:
            self.source_mapping.to_csv(self.source_mapping_path)

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
            split_text = text.split("\n", 1)
            if len(split_text) == 1:
                continue
            text = split_text[1]  # Remove heading
        elif len(headings) == 0:
            heading = ""
        else:
            continue

        final_dict[heading] = str(text)

    return final_dict


class PageReporter:
    last_page: int = 0

    @classmethod
    def report(cls, page: int) -> None:
        if page == cls.last_page:
            return

        cls.last_page = page

        if page % 2500 == 0:
            print(f"{page:6d}")
        elif page % 100 == 0:
            print(f"{page:6d}", end=".")


def prepare_wiki(collection_name: str, wiki_path: str, output_dir: str) -> list[str]:
    print(f"Parsing wiki from file {wiki_path}")

    file_num = 1

    ctx = WikiParseContext()
    ctx.collection_name = collection_name
    ctx.output_dir = output_dir
    passage_files: List[str] = [ctx.reset(file_num)]

    for event, elem in ET.iterparse(wiki_path, events=("start", "start-ns", "end")):
        if event == "start-ns":
            print("new namespace", elem)
            if elem[0] == "":
                print(f"set namespace from {ctx.namespace} to {elem[1]}")
                ctx.namespace = elem[1]
            continue

        if elem.tag == ctx.page_tag:
            if event == "start":
                ctx.new_page()
                PageReporter.report(ctx.pushed)
        elif elem.tag == ctx.ns_tag:
            if event == "end" and elem.text != "0":
                ctx.should_parse = False
        elif elem.tag == ctx.redirect_tag:
            ctx.should_parse = False
        elif elem.tag == ctx.title_tag:
            if event == "end":
                ctx.current_title = elem.text
        elif elem.tag == ctx.text_tag:
            if (
                event == "end"
                and ctx.should_parse
                and not is_title_banned(ctx.current_title)
            ):
                ctx.current_page = parse_wikitext(elem.text)

        if ctx.pushed > 25000:
            print(
                f"==== pushed {ctx.pushed} passages, creating new file ====", flush=True
            )
            file_num += 1
            passage_files.append(ctx.reset(file_num))

    ctx.push()
    ctx.flush()
    return passage_files


class PrepareWiki(BaseNode):

    def __init__(self, name: str, output_dir: str):
        super().__init__(name, [list], ListDescriptor())
        self.output_dir = output_dir

    def process(self, inp: list[WikiDataFile]) -> list[str]:
        passage_files: List[str] = list()
        for file in inp:
            pf = prepare_wiki(
                f"wiki-{file.num}-p{file.p_first}-p{file.p_last}",
                file.path,
                self.output_dir,
            )
            passage_files += pf

        return passage_files


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python -m colbert_search prepare_wiki [collection_name] [wiki_file] [output_dir]"
        )
        exit(1)

    prepare_wiki(*sys.argv[1:])
