from __future__ import annotations

import re
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


class _FuncWrapper:
    """
    Wrapper that allows 'or' operation on functions for chaining function calls

    Examples
    --------
    Given wrapped functions ``f1``, ``f2`` and ``f3``
    >>> r = (f1 | f2 | f3)("hello")

    is Equal to

    >>> r = f3(f2(f1("hello")))

    """

    def __init__(self, func: Callable[[str], str]):
        self._func = func
        self.__name__ = func.__name__

    def __call__(self, text: str) -> str:
        return self._func(text)

    @staticmethod
    def _wrap_or(lhs: Callable[[str], str], rhs: Callable[[str], str]) -> Callable[[str], str]:
        @_FuncWrapper
        def _wrapper(text: str) -> str:
            return rhs(lhs(text))

        return _wrapper

    def __or__(self, other: _FuncWrapper | Callable[[str], str]):
        if not isinstance(other, _FuncWrapper):
            if not callable(other):
                return NotImplemented

            return _FuncWrapper._wrap_or(self._func, other)

        return _FuncWrapper._wrap_or(self._func, other._func)

    def __ror__(self, other: Callable[[str], str]):
        # other is not a _FuncWrapper, because __or__ would be called
        if not callable(other):
            return NotImplemented

        return _FuncWrapper._wrap_or(other, self._func)


@_FuncWrapper
def remove_punctuation(text: str) -> str:
    def pred(i: int):
        ch = chr(i)
        allowed = "!%'.:;?"
        if ch in allowed:
            return False
        cat = unicodedata.category(ch)
        return cat == "Pf" or cat == "Pi"

    transform_table = {i: None for i in range(sys.maxunicode) if pred(i)}

    return text.translate(transform_table)


_wiki_formatting_regex = re.compile(r"((\w+\|)+\w+|'''|'')")


@_FuncWrapper
def remove_wiki_formatting(text: str) -> str:
    return re.sub(_wiki_formatting_regex, "", text)


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
        return cls(name, func, [str], StrDescriptor())  # type: ignore

    @classmethod
    def new_for_lists(cls, name: str, func: Callable[[str], str]) -> TextProcessingNode[List[str]]:
        def wrapper(batch: List[str]) -> List[str]:
            return [func(b) for b in batch]

        return cls(name, wrapper, [list], ListDescriptor())  # type: ignore

    @classmethod
    def new_for_dicts(cls, name: str, func: Callable[[str], str]) -> TextProcessingNode[Dict[str, str]]:
        def wrapper(batch: Dict[str, str]) -> Dict[str, str]:
            return {k: func(v) for k, v in batch.items()}

        return cls(name, wrapper, [dict], DictDescriptor())  # type: ignore

    def process(self, inp: T) -> T:
        return self.func(inp)
