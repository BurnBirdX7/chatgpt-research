from __future__ import annotations

import typing as t
import os

from src import SourceMapping

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

    def new_passages(self, title: str, sections: t.Dict[str, t.List[str]]):
        safe_title = title.replace('-', '_').replace(' ', '_')
        for heading, section in sections.items():
            if is_title_banned(heading):
                continue

            if "\n" in heading.strip():  # Header is malformed, remove part of it
                heading = heading.strip().split("\n", maxsplit=1)[0]
            safe_header = heading.replace('-', '_').replace(' ', '_')

            section_start_pid = self.pid
            for paragraph in section:
                safe_paragraph = paragraph.replace("\t", " ")  # make tsv-safe
                self.passage_file.write(f"{self.pid}\t{safe_paragraph}\n")
                self.pid += 1

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