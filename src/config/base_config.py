"""
Base Config and help classes

Idea: https://stackoverflow.com/a/58081120
"""

from __future__ import annotations

__all__ = ["BaseConfig"]

import os
import dataclasses
from typing import Any, TypeVar, Type

T = TypeVar("T")


@dataclasses.dataclass
class BaseConfig:
    def get(self, key: str) -> Any:
        return getattr(self, key)

    def is_config(self, cls: type) -> bool:
        return isinstance(self, cls)

    @classmethod
    def load_from_env(cls: Type["BaseConfig"], use_defaults: bool = True, prefix: str | None = None):
        """
        Loads config fields from environment variables

        Examples
        --------
        For class named WikiServerConfig and its fields ``ip_address`` and ``port``,
        environment variables ``WIKISERVERCONFIG_IP_ADDRESS`` and ``WIKISERVERCONFIG_PORT`` will be accessed
        (if other `prefix` isn't provided)

        if prefix is provided, names will be {PREFIX}_IP_ADDRESS and {PREFIX}_PORT

        Parameters
        ----------
        cls : type
            class that is being instantiated

        use_defaults : bool, default = True
            whether to use default values or load everything from environment

        prefix : str, optional
            alternative prefix
        """

        if prefix is None:
            prefix = cls.__name__.upper()

        d = dict()
        for field in dataclasses.fields(cls):
            env_name = prefix + "_" + field.name.upper()
            if env_name not in os.environ:
                if not use_defaults:
                    raise ValueError(f"Environment variable {env_name} is required but not set")
            else:
                d[field.name] = os.environ[env_name]

        return cls.__call__(**d)
