from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Callable

from src.pipeline.BaseDataDescriptor import BaseDataDescriptor

InT = TypeVar('InT')
OutT = TypeVar('OutT')


class Block(Generic[InT, OutT], ABC):

    def __init__(self,
                 name: str,
                 in_type: type[InT],
                 out_descriptor: BaseDataDescriptor[OutT]):

        if '$' in name:
            raise ValueError('Prohibited character `$` in block name')

        self.__name = name
        self.in_type = in_type

        out_descriptor.block_name = name
        self.out_descriptor = out_descriptor
        self.out_type = self.out_descriptor.get_data_type()

    def set_artifacts_folder(self, artifacts_folder: str):
        self.out_descriptor.artifacts_folder = artifacts_folder

    def is_type_acceptable(self, typ: type) -> bool:
        return issubclass(typ, self.in_type)

    def is_value_acceptable(self, val: object) -> bool:
        return isinstance(val, self.in_type)

    def __repr__(self):
        return f'{self.__class__.__name__}(\"{self.name}\" [{self.in_type.__name__} -> {self.out_type.__name__}])'

    @property
    def name(self) -> str:
        return self.__name

    @abstractmethod
    def process(self, inp: InT) -> OutT:
        """
        Implement block logic in this method
        """
        raise NotImplemented

