from __future__ import annotations

__all__ = ["map_block"]

import inspect
from typing import Callable, TypeVar, List, cast, get_type_hints

from .base_data_descriptor import BaseDataDescriptor
from .nodes import Node, BaseNode

OutT = TypeVar('OutT')


def map_block(
        out_descriptor: BaseDataDescriptor[OutT],
) -> Callable[[Callable], type[Node]]:
    r"""
    Decorator that turns mapping function into Block for a pipeline

    Usage:
    @map_block(StrDescriptor())
    def LongString(a: int) -> str:
        return f'{a}{a}{a}'

    Decorated function MUST be annotated with *specific* types,
        These types are used for type-checking in runtime
        (Parametrized and generics won't work properly - use dict instead of dict[str, str])

    You can provide type_hint for input type in @map_block(desc, in_type_hint)
    """

    def decorator(func: Callable) -> type[BaseNode]:
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        in_types: List[type] = []
        for param in sig.parameters.keys():
            if param not in type_hints:
                raise TypeError(f"Parameter {param} of function {func.__name__} doesn't have a type hint")
            in_types.append(type_hints[param])

        if 'return' not in type_hints:
            raise TypeError(f"Function {func.__name__} does not have a return type annotation")

        if not issubclass(type_hints['return'], out_descriptor.get_data_type()):
            raise TypeError(f"Return type descriptor for function {func.__name__}"
                            f"is incompatible with its annotated return type")

        for typ in in_types:
            if not isinstance(typ, type):
                raise TypeError(f"Expected a type but got {typ!r} [{type(typ)}]")

        class WrapperNode(BaseNode):
            def __init__(self, name: str):
                super().__init__(name, in_types, out_descriptor)
                self._wrapped = func

            def process(self, *inp) -> OutT:
                return self._wrapped(*inp)

        WrapperNode.__name__ = func.__name__
        return WrapperNode

    return cast(Callable, decorator)
