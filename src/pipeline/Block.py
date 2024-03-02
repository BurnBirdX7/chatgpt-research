from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Callable

from src.pipeline.BaseDataDescriptor import BaseDataDescriptor

InT = TypeVar('InT')
OutT = TypeVar('OutT')


class Block(Generic[InT, OutT], ABC):

    def __init__(self,
                 block_name: str,
                 in_descriptor_type: type[BaseDataDescriptor[InT]],
                 out_descriptor_type: type[BaseDataDescriptor[OutT]],
                 artifacts_path: str = "pipe-artifacts"):

        self.block_name = block_name
        self.out_descriptor = out_descriptor_type(block_name, artifacts_path)
        self.in_descriptor_type = in_descriptor_type
        self.out_descriptor_type = out_descriptor_type

    @abstractmethod
    def process(self, inp: InT) -> OutT:
        raise NotImplemented

