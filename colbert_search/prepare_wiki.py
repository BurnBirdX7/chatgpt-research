from __future__ import annotations

import argparse
import html
import logging
import re
import sys
import typing as t
import xml.etree.ElementTree as ET
from collections import defaultdict

import mwparserfromhell as mw

from _collection_context import CollectionContext, is_title_banned, SplitCollectionContext
from src.pipeline import BaseNode, ListDescriptor
from src import WikiDataFile

from _xml_parse_runner import XmlParseRunner


def _not_table(node: mw.nodes.Node) -> bool:
    # Tables are excluded from survey
    if isinstance(node, mw.nodes.Tag):
        if node.tag == "table":
            return False
    return True


def _not_ref(node: mw.nodes.Node) -> bool:
    # Reference links are excluded from survey
    return not re.match(r"(<ref.*>.*</ref>|<ref .*/>)", str(node))


def _not_special_link(node: mw.nodes.Node) -> bool:
    # Special links are excluded due to parsing problems
    return not re.match(r"\[\[(File|Help|Extention|User|Manual):.*]]", str(node))


def _not_empty(node: mw.nodes.Node) -> bool:
    # Nodes that cannot be rendered as text, or nodes that consist of whitespace chars are omitted
    s = node.__strip__()
    return s is not None and s.strip() != ""


def _ok(node: mw.nodes.Node) -> bool:
    return _not_table(node) and _not_ref(node) and _not_special_link(node) and _not_empty(node)


def parse_wikitext(text: str) -> t.Dict[str, t.List[str]]:
    """
    Transforms string with WikiText into dictionary with section names as keys and lists of paragraphs as values

    """
    result = defaultdict(list)
    code = mw.parse(text)
    sections = code.get_sections(flat=True)
    section: mw.wikicode.Wikicode
    for i, section in enumerate(sections):
        if len(section.nodes) == 0:
            # Empty section
            continue

        heading: str
        if isinstance(section.get(0), mw.nodes.Heading):
            nodes = section.nodes[1:]
            heading = "".join(map(str, section.get(0).title.ifilter_text(recursive=False))).strip()
        else:
            nodes = section.nodes
            heading = ""

        # Collect full text
        section_text = ""
        for node in nodes:
            if _ok(node):
                section_text += html.unescape(str(node.__strip__()))

        # Split into paragraphs
        for p in section_text.split("\n"):
            if len(p.strip()) < 10:
                continue
            result[heading].append(p.strip())

    return result


class Reporter:
    def __init__(self, logger: logging.Logger):
        self.rate: int = 100
        self.next_limit = self.rate
        self.logger = logger
        self.page_count = 0
        self.passage_count = 0

    def report(self, passages: int) -> None:
        self.page_count += 1
        self.passage_count += passages
        if self.page_count < self.next_limit:
            return

        self.next_limit += self.rate
        self.logger.debug(f"Processed {self.page_count} pages and pr oduced {self.passage_count} passages...")

def prepare_wiki(
    collection_name: str, wiki_path: str, output_dir: str, logger: logging.Logger = logging.getLogger(__name__)
) -> t.List[CollectionContext.Files]:
    logger.info(f"Parsing wiki XML, path: \"{wiki_path}\"")
    reporter = Reporter(logger)

    def _page_process(elem: ET.Element):
        nonlocal reporter
        if "ns" not in data or data["ns"] != "0":
            return
        if "redirect" in data:
            return
        if "title" not in data or is_title_banned(data["title"]):
            return

        sections = parse_wikitext(elem.text)
        passage_count = collection_ctx.last().new_passages(data["title"], sections)
        reporter.report(passage_count)

    def _clear_data(_):
        data.clear()

    def _update_context(_):
        nonlocal collection_ctx
        if collection_ctx.last_ctx.pid >= 1_000_000:
            collection_ctx.new()

    collection_ctx = SplitCollectionContext(collection_name, output_dir, logger)
    collection_ctx.new()

    data: t.Dict[str, str] = {}
    parse_ctx = XmlParseRunner(wiki_path)
    parse_ctx.collect_contents("ns", data)
    parse_ctx.collect_contents("redirect", data)
    parse_ctx.collect_contents("title", data)

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
                self.logger,
            )
            passage_files += pf

        return passage_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "python -m colbert_search prepare_wiki", description="Prepare number of collections from a single wiki dump"
    )

    parser.add_argument(
        "--collection", "-c", action="store", dest="collection_name", help="name of produced collection", required=True
    )
    parser.add_argument("--file", "-f", action="store", dest="wiki_path", help="path to the wiki .xml", required=True)
    parser.add_argument(
        "--output",
        "-o",
        action="store",
        dest="output_dir",
        help="directory where new collections will be dumped",
        required=True,
    )
    namespace = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    prepare_wiki(**vars(namespace))
