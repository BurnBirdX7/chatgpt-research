"""
Base Config and help classes

Idea: https://stackoverflow.com/a/58081120
"""

from dataclasses import dataclass, fields
from typing import Any, TypeVar, Generic, Union

T = TypeVar("T")


@dataclass
class DefaultValue(Generic[T]):
    val: T


@dataclass
class BaseConfig:
    def __post_init__(self: "BaseConfig"):
        for field in fields(self):
            if isinstance(field.default, DefaultValue):
                val = getattr(self, field.name)
                if isinstance(val, DefaultValue) or val is None:
                    setattr(self, field.name, field.default.val)

    def get(self, key: str) -> Any:
        return getattr(self, key)

    def is_config(self, cls: type) -> bool:
        return isinstance(self, cls)
