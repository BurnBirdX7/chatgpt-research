from __future__ import annotations

import inspect
from abc import ABC
from typing import Callable, TypeVar, cast

from .BaseDataDescriptor import BaseDataDescriptor
from .Block import Block

InT = TypeVar('InT')
OutT = TypeVar('OutT')

class IBlock(Block[InT, OutT], ABC):
    """
    An abstract class used for typing purposes
    """
    def __init__(self, name: str):
        super().__init__(name, None, None)  # type: ignore
        raise NotImplementedError("You shouldn't inherit from this class nor create any instances of it")


def map_block(
        descriptor: BaseDataDescriptor[OutT],
        in_type_hint: type[InT] | None = None,
) -> Callable[[Callable[[InT], OutT]], type[IBlock[InT, OutT]]]:
    r"""
    Decorator that turns mapping function into Block for a pipeline

    Usage:
    @map_block(StrDescriptor())
    def LongString(a: int) -> str:
        return f'{a}{a}{a}'

    Decorated function MUST be type-annotated,
        these annotations are source for Block types

    You can provide type_hint for input type in @map_block(desc, in_type_hint)
    """

    def decorator(func: Callable[[InT], OutT]) -> type[Block[InT, OutT]]:
        sig = inspect.signature(func)
        if len(sig.parameters) != 1:
            raise TypeError("map_block only accepts functions with one argument")

        param_sig = list(sig.parameters.values())[0]
        in_type = param_sig.annotation
        out_type = sig.return_annotation

        if in_type_hint is not None and not issubclass(in_type, in_type_hint):
            raise TypeError(f"Provided type_hint does not match")

        if param_sig.name not in func.__annotations__:
            raise TypeError(f"Parameter {param_sig.name} is not type-annotated")

        if 'return' not in func.__annotations__:
            raise TypeError(f"Return type is not annotated")

        if not descriptor.is_type_compatible(out_type):
            raise TypeError(f"{out_type} cannot be used with descriptor {descriptor}")

        class WrapperBlock(Block[InT, OutT]):
            def __init__(self, name: str):
                super().__init__(name, in_type, descriptor)
                self._wrapped = func

            def process(self, inp: InT) -> OutT:
                return self._wrapped(inp)

        WrapperBlock.__name__ = func.__name__
        return WrapperBlock

    return cast(Callable, decorator)


T = TypeVar('T')

class NoopBlock(Block[T, T]):
    def __init__(self, name: str, descriptor: BaseDataDescriptor[T]):
        super().__init__(name, descriptor.get_data_type(), descriptor)

    def process(self, inp: T) -> T:
        return inp
