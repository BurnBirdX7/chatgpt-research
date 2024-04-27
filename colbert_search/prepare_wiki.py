from __future__ import annotations

import os.path
import sys
import typing as t
import urllib.parse
import xml.etree.ElementTree as ET
import mwparserfromhell as mw

from src.pipeline import BaseNode, ListDescriptor
from src import WikiDataFile, SourceMapping

from parse_context import ParseContext

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

def parse_wikitext(text: str) -> t.Dict[str, str]:
    """
    :param text: WikiText
    :return: Dictionary of sections, keys are section names and values are section texts
    """
    final_dict = {}
    wikicode = mw.parse(text)
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


class SplitCollectionContext:
    def __init__(self, collection_name: str, path: str):
        self.collection_name = collection_name
        self.fid: int = 0
        self.path = path
        self.past_cts: t.List[CollectionContext] = []
        self.last_ctx: CollectionContext | None = None

    def new(self) -> CollectionContext:
        self.fid += 1
        self.close()
        self.last_ctx = CollectionContext(self.collection_name, self.fid, self.path)
        return self.last_ctx

    def last(self) -> CollectionContext | None:
        return self.last_ctx

    def close(self):
        if self.last_ctx is None:
            return
        self.last_ctx.close()
        self.past_cts.append(self.last_ctx)
        self.last_ctx = None


class CollectionContext:
    class Files(t.NamedTuple):
        passage_file_path: str
        mapping_file_path: str

    def __init__(self, collection_name: str, num: int, path: str):
        self.collection_name = collection_name
        self.path = path

        if not os.path.exists(path):
            os.makedirs(path)

        self.passage_file_path = os.path.join(self.path, f"{self.collection_name}_passages_{num}.tsv")
        self.mapping_file_path = os.path.join(self.path, f"{self.collection_name}_sources_{num}.csv")

        self.passage_file = open(self.passage_file_path, "w")
        self.mapping = SourceMapping()

        self.pid = 0
        self.closed = False

    def new_passages(self, title: str, sections: t.Dict[str, str]):
        for heading, section in sections.items():
            if is_title_banned(heading):
                continue

            if "\n" in heading.strip():  # Header is malformed, remove part of it
                heading = heading.strip().split("\n", maxsplit=1)[0]
            safe_header = urllib.parse.quote_plus(heading)

            section_start_pid = self.pid
            for paragraph in section.split("\n"):
                safe_paragraph = paragraph.strip()
                if len(safe_paragraph) == 0:
                    continue

                safe_paragraph = safe_paragraph.replace("\t", " ")  # make tsv-safe
                self.passage_file.write(f"{self.pid}\t{safe_paragraph}\n")
                self.pid += 1

            safe_title = urllib.parse.quote_plus(title)
            url = f"https://en.wikipedia.org/wiki/{safe_title}#{safe_header}"
            self.mapping.append_interval(self.pid - section_start_pid, url)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.passage_file.close()
        self.mapping.to_csv(self.mapping_file_path)

    def get_files(self) -> Files:
        return self.Files(self.passage_file_path, self.mapping_file_path)


class Reporter:
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


def prepare_wiki(collection_name: str, path: str, output_dir: str) -> t.List[CollectionContext.Files]:
    def _page_process(elem: ET.Element):
        if "ns" not in data or data["ns"] != "0":
            return
        if "redirect" in data:
            return
        if "title" not in data or is_title_banned(data["title"]):
            return

        sections = parse_wikitext(elem.text)
        collection_ctx.last().new_passages(data["title"], sections)

    a = False

    def _update_context(_):
        nonlocal collection_ctx
        if collection_ctx.last_ctx.pid >= 100:
            collection_ctx.new()
            nonlocal a
            if a:
                raise StopIteration
            a = True

    print(f"Parsing wiki from file {path}")

    collection_ctx = SplitCollectionContext(collection_name, output_dir)
    collection_ctx.new()

    data: t.Dict[str, str] = {}
    parse_ctx = ParseContext(path)
    parse_ctx.collect_contents("ns", data)
    parse_ctx.collect_contents("redirect", data)
    parse_ctx.collect_contents("title", data)

    def _clear_data(_):
        data.clear()

    parse_ctx.end_handler("text", _page_process)
    parse_ctx.end_handler("page", _clear_data)
    parse_ctx.end_handler("page", _update_context)
    parse_ctx.run()

    collection_ctx.close()

    return [ctx.get_files() for ctx in collection_ctx.past_cts]


class PrepareWiki(BaseNode):

    def __init__(self, name: str, output_dir: str):
        super().__init__(name, [list], ListDescriptor())
        self.output_dir = output_dir

    def process(self, inp: t.List[WikiDataFile]) -> list[str]:
        passage_files: t.List[str] = list()
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
        print("Usage: python -m colbert_search prepare_wiki [collection_name] [wiki_file] [output_dir]")
        exit(1)

    prepare_wiki(*sys.argv[1:])
