from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from typing import Any

from src.pipeline import BaseDataDescriptor, map_block
from src.pipeline.BaseDataDescriptor import Value


@dataclass
class WikiFile:
    """
    Describes file that contains data from wikipedia
    """

    path: str     # name of the file
    date: str
    num: int
    p_first: int  # first page ID present in file
    p_last: int   # last page ID present in file

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "WikiFile":
        return WikiFile(
            d["name"],
            d["date"],
            int(d["num"]),
            int(d["first_id"]),
            int(d["last_id"])
        )

    def __lt__(self, other: "WikiFile"):
        if self.date < other.date:
            return True
        if self.date == other.date:
            if self.num < other.num:
                return True
            if self.num == other.num:
                return self.p_first < self.p_last

        return False


class WikiBz2FileDescriptor(BaseDataDescriptor[list[WikiFile]]):
    def store(self, data: list[WikiFile]) -> dict[str, Value]:
        return {
            'list': [
                asdict(bz2file)
                for bz2file in data
            ]
        }

    def load(self, dic: dict[str, Value]) -> list[WikiFile]:
        return [
            WikiFile.from_dict(bz2dict)
            for bz2dict in dic['list']
        ]

    def get_data_type(self) -> type:
        return list

    def is_type_compatible(self, typ: type | None):
        return issubclass(typ, list) or typ == list[WikiFile]


bz2_file_pattern = re.compile(r"^enwiki-(\d{8})-pages-articles-multistream(\d+).xml-p(\d+)p(\d+)\.bz2$")

@map_block(WikiBz2FileDescriptor(), str)
def FindBz2Files(path: str) -> list[WikiFile]:
    dir_files = [
        file
        for file in os.listdir(path)
        if os.path.isfile(os.path.join(path, file))
    ]
    bz2_files = list[WikiFile]()

    for file in dir_files:
        match = bz2_file_pattern.match(file)
        if match is None:
            continue

        date, num, p_first, p_last = match.groups()
        bz2_files.append(WikiFile(
            os.path.join(path, file), date, int(num), int(p_first), int(p_last)
        ))

    bz2_files.sort()
    return bz2_files
