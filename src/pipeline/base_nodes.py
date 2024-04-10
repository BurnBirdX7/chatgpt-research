from __future__ import annotations

__all__ = ['BaseNode', 'Node']

import logging
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

from src.pipeline.base_data_descriptor import BaseDataDescriptor


class Node(ABC):
    """
    Interface for nodes, inherit BaseNode to implement your own node
    """

    def __init__(self, name: str):
        if '$' in name:
            raise ValueError('Prohibited character `$` in block name')

        self.__logger = logging.getLogger(f"node.{self.__class__.__name__}.{name}")
        self.__name = name

    @property
    def logger(self):
        return self.__logger

    @logger.setter
    def logger(self, new_logger: logging.Logger):
        self.__logger = new_logger

    @property
    def name(self) -> str:
        return self.__name

    @property
    @abstractmethod
    def in_types(self) -> List[type]:
        ...

    @property
    @abstractmethod
    def out_type(self) -> type:
        ...

    @property
    @abstractmethod
    def out_descriptor(self) -> BaseDataDescriptor:
        ...

    @abstractmethod
    def set_artifacts_folder(self, artifacts_folder: str):
        ...

    @abstractmethod
    def is_type_acceptable(self, input_num: int, typ: type) -> bool:
        ...

    @abstractmethod
    def is_value_acceptable(self, input_num: int, value: Any) -> bool:
        ...

    @abstractmethod
    def is_input_type_acceptable(self, typs: List[type] | Tuple[type]) -> bool:
        ...

    @abstractmethod
    def __repr__(self):
        ...

    @abstractmethod
    def process(self, *inp) -> Any:
        ...

    def prerequisite_check(self) -> str | None:
        """
        Override this method to check if the prerequisites are met BEFORE running the pipeline

        :return: None if everything is ok, or error if prerequisites are not met
        """
        return None


class BaseNode(Node, ABC):

    def __init__(self,
                 name: str,
                 in_types: List[type] | Tuple[type],
                 out_descriptor: BaseDataDescriptor):
        super().__init__(name)

        # Input:
        self.__in_types = list(in_types)

        # Output:
        out_descriptor.block_name = name
        self.__out_descriptor = out_descriptor
        self.__out_type = self.out_descriptor.get_data_type()

        # Misc:
        self.out_descriptor.logger = logging.getLogger(f"{self.logger.name}.{self.out_descriptor.__class__.__name__}")

    @Node.logger.setter
    def logger(self, new_logger):
        Node.logger.fset(self, new_logger)
        self.out_descriptor.logger = logging.getLogger(f"{self.logger.name}.{self.out_descriptor.__class__.__name__}")

    @property
    def in_types(self) -> List[type]:
        return self.__in_types

    @property
    def out_type(self) -> type:
        return self.__out_type

    @property
    def out_descriptor(self) -> BaseDataDescriptor:
        return self.__out_descriptor

    def set_artifacts_folder(self, artifacts_folder: str):
        self.out_descriptor.artifacts_folder = artifacts_folder

    def is_type_acceptable(self, input_num: int, typ: type) -> bool:
        return issubclass(typ, self.in_types[input_num])

    def is_value_acceptable(self, input_num: int, value: Any) -> bool:
        return self.is_type_acceptable(input_num, type(value))

    def is_input_type_acceptable(self, typs: List[type] | Tuple[type]) -> bool:
        if len(typs) != len(self.in_types):
            return False
        for i, typ in enumerate(typs):
            if not self.is_type_acceptable(i, typ):
                return False

        return True

    def __repr__(self):
        typs = ','.join([typ.__name__ for typ in self.in_types])
        return f'{self.__class__.__name__}(\"{self.name}\" [({typs}) -> {self.out_type.__name__}])'

    @abstractmethod
    def process(self, *inp) -> Any:
        """
        Implement block logic in this method
        """


class ConstantNode(BaseNode):
    def __init__(self, name: str, value: Any, out_descriptor: BaseDataDescriptor):
        super().__init__(name, [], out_descriptor)
        self.value = value
        if not out_descriptor.is_type_compatible(type(value)):
            raise TypeError(f"Value {value} of type {type(value)} is not compatible with descriptor {out_descriptor!r}")

    def process(self, *ignored) -> Any:
        return self.value

