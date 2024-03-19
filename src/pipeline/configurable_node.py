from __future__ import annotations

__all__ = ['ConfigurableNode']

from abc import abstractmethod, ABC
from typing import Type, TypeVar, Generic, Tuple

from .base_data_descriptor import BaseDataDescriptor
from .nodes import BaseNode
from src.config import BaseConfig

InT = TypeVar("InT")
OutT = TypeVar("OutT")
ConfigT = TypeVar("ConfigT", bound=BaseConfig)

class ConfigurableNode(Generic[InT, OutT, ConfigT], BaseNode, ABC):

    def __init__(self,
                 name: str,
                 in_type: Type[InT],
                 out_descriptor: BaseDataDescriptor[OutT]):
        super().__init__(name, in_type, out_descriptor)
        self.config: ConfigT | None = None

    @abstractmethod
    def get_config_spec(self) -> Tuple[str, type[ConfigT]]:
        """
        Return config name and expected type
        """
        pass

    @abstractmethod
    def get_default_config(self) -> BaseConfig:
        """
        Returns default config
        """
        pass

    def set_config(self, config: ConfigT) -> None:
        self.config = config
