from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, List, Dict

from src.pipeline import BaseDataDescriptor
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
    def from_dict(d: Dict[str, Any]) -> "WikiFile":
        return WikiFile(
            d["path"],
            d["date"],
            int(d["num"]),
            int(d["p_first"]),
            int(d["p_last"])
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


class ListWikiFileDescriptor(BaseDataDescriptor[List[WikiFile]]):
    def store(self, data: List[WikiFile]) -> Dict[str, Value]:
        return {
            'list': [
                asdict(bz2file)
                for bz2file in data
            ]
        }

    def load(self, dic: Dict[str, Value]) -> List[WikiFile]:
        return [
            WikiFile.from_dict(bz2dict)
            for bz2dict in dic['list']
        ]

    def get_data_type(self) -> type:
        return list

    def is_type_compatible(self, typ: type | None):
        return issubclass(typ, list) or typ == List[WikiFile]