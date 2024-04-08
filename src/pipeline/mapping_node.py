from __future__ import annotations

__all__ = ["mapping_node"]

import inspect
from typing import Callable, TypeVar, List, cast, get_type_hints

from .base_data_descriptor import BaseDataDescriptor
from .nodes import Node, BaseNode

OutT = TypeVar('OutT')


def mapping_node(
        out_descriptor: BaseDataDescriptor[OutT],
) -> Callable[[Callable], type[Node]]:
    r"""
    Parametrized decorator that turns mapping function into a Node for a pipeline

    Parameters
    ----------
    out_descriptor : BaseDataDescriptor
        Data descriptor for result of the decorated function

    Examples
    --------
    >>> @mapping_node(StrDescriptor())
    ... def LongString(a: int) -> str:
    ...     return f'{a}{a}{a}'

    Notes
    -----
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

        if not out_descriptor.is_type_compatible(type_hints['return']):
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
        WrapperNode.__doc__ = f"Node subclass that wrapps a function. See, doc for the `process` method"
        WrapperNode.process.__doc__ = func.__doc__
        return WrapperNode

    return cast(Callable, decorator)
