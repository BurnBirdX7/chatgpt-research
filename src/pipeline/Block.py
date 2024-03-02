from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Callable

from src.pipeline.BaseDataDescriptor import BaseDataDescriptor

InT = TypeVar('InT')
OutT = TypeVar('OutT')


class Block(Generic[InT, OutT], ABC):

    def __init__(self,
                 block_name: str,
                 in_type: type[InT],
                 out_descriptor: BaseDataDescriptor[OutT]):

        self.name = block_name
        self.in_type = in_type

        out_descriptor.block_name = block_name
        self.out_descriptor = out_descriptor
        self.out_type = self.out_descriptor.get_data_type()

    def set_artifacts_folder(self, artifacts_folder: str):
        self.out_descriptor.artifacts_folder = artifacts_folder

    @abstractmethod
    def process(self, inp: InT) -> OutT:
        """
        Implement block logic in this method
        """
        raise NotImplemented

