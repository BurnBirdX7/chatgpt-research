from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, List, Dict, cast

from src.pipeline import BaseDataDescriptor
from src.pipeline.base_data_descriptor import ValueType


@dataclass
class WikiDataFile:
    """
    Describes file that contains data from wikipedia
    """

    path: str  # name of the file
    date: str
    num: int
    p_first: int  # first page ID present in file
    p_last: int  # last page ID present in file

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "WikiDataFile":
        return WikiDataFile(
            d["path"], d["date"], int(d["num"]), int(d["p_first"]), int(d["p_last"])
        )

    def __lt__(self, other: "WikiDataFile"):
        if self.date < other.date:
            return True
        if self.date == other.date:
            if self.num < other.num:
                return True
            if self.num == other.num:
                return self.p_first < self.p_last

        return False


class ListWikiFileDescriptor(BaseDataDescriptor[List[WikiDataFile]]):
    def store(self, data: List[WikiDataFile]) -> Dict[str, ValueType]:
        return {"list": [asdict(bz2file) for bz2file in data]}

    def load(self, dic: Dict[str, ValueType]) -> List[WikiDataFile]:
        return [
            WikiDataFile.from_dict(bz2dict)
            for bz2dict in cast(List[Dict[str, Any]], dic["list"])
        ]

    def get_data_type(self) -> type:
        return list

    def is_type_compatible(self, typ: type | None):
        return typ is not None and issubclass(typ, list) or typ == List[WikiDataFile]
