"""
Base Config and help classes

Idea: https://stackoverflow.com/a/58081120
"""
from __future__ import annotations

__all__ = ["BaseConfig", "DefaultValue"]

import os
from dataclasses import dataclass, fields
from typing import Any, TypeVar, Generic, Union, Type

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
                if isinstance(val, DefaultValue):
                    setattr(self, field.name, field.default.val)

    def get(self, key: str) -> Any:
        return getattr(self, key)

    def is_config(self, cls: type) -> bool:
        return isinstance(self, cls)

    @classmethod
    def load_from_env(cls: Type["BaseConfig"], use_defaults: bool = True, prefix: str | None = None) -> "BaseConfig":
        """
        Loads config fields from environment variables

        For class named WikiServerConfig and its fields `ip_address` and `port`,
        environment variables WIKISERVERCONFIG_IP_ADDRESS and WIKISERVERCONFIG_PORT will be accessed
        (if other prefix isn't provided)

        if prefix is provided, names will be {PREFIX}_IP_ADDRESS and {PREFIX}_PORT

        :param cls: class that is being instantiated
        :param use_defaults: whether to use default values or load everything from environment
        :param prefix: alternative prefix
        """

        if prefix is None:
            prefix = cls.__name__.upper()

        d = dict()
        for field in fields(cls):
            env_name = prefix + "_" + field.name.upper()
            if env_name not in os.environ:
                if not use_defaults:
                    raise ValueError(f"Environment variable {env_name} is required but not set")
            else:
                d[field.name] = os.environ[env_name]

        return cls.__call__(**d)
