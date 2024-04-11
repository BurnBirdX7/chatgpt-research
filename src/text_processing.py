from __future__ import annotations

import sys
import typing
import unicodedata
from typing import Callable, List, Generic, Union, Dict

from .pipeline import (
    BaseNode,
    BaseDataDescriptor,
    StrDescriptor,
    ListDescriptor,
    DictDescriptor,
)


def remove_punctuation(text: str) -> str:
    def pred(i: int):
        ch = chr(i)
        allowed = "!%'.:;?"
        if ch in allowed:
            return False
        cat = unicodedata.category(ch)
        return cat.startswith("P")

    transform_table = {i: None for i in range(sys.maxunicode) if pred(i)}

    return text.translate(transform_table)


T = typing.TypeVar("T", bound=Union[str, List[str], Dict[str, str]])


class TextProcessingNode(BaseNode, Generic[T]):
    def __init__(
        self,
        name: str,
        func: Callable[[T], T],
        in_types: List[type],
        out_descriptor: BaseDataDescriptor,
    ):
        super().__init__(name, in_types, out_descriptor)
        self.func = func

    @classmethod
    def new(cls, name: str, func: Callable[[str], str]) -> TextProcessingNode[str]:
        return cls(name, func, [str], StrDescriptor())

    @classmethod
    def new_for_lists(
        cls, name: str, func: Callable[[str], str]
    ) -> TextProcessingNode[List[str]]:
        def wrapper(batch: List[str]) -> List[str]:
            return [func(b) for b in batch]

        return cls(name, wrapper, [list], ListDescriptor())

    @classmethod
    def new_for_dicts(
        cls, name: str, func: Callable[[str], str]
    ) -> TextProcessingNode[Dict[str, str]]:
        def wrapper(batch: Dict[str, str]) -> Dict[str, str]:
            return {k: func(v) for k, v in batch.items()}

        return cls(name, wrapper, [dict], DictDescriptor())

    def process(self, inp: T) -> T:
        return self.func(inp)
