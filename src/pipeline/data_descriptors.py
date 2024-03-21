from __future__ import annotations

__all__ = ['EmptyDataDescriptor',
           'InDictDescriptor',
           'IntDescriptor',
           'FloatDescriptor',
           'BytesDescriptor',
           'StrDescriptor',
           'ListDescriptor',
           'DictDescriptor']

import os.path
from abc import ABC
from typing import TypeVar, cast, List, Dict, Type

from .base_data_descriptor import BaseDataDescriptor, ValueType

T = TypeVar('T')

class EmptyDataDescriptor(BaseDataDescriptor[None]):
    @classmethod
    def get_data_type(cls) -> Type[None]:
        return type(None)

    def store(self, data: None) -> Dict[str, ValueType]:
        return dict()

    def load(self, dic: Dict[str, ValueType]) -> None:
        return None

    def is_type_compatible(self, typ: type | None):
        return typ is None or issubclass(typ, type(None))


class InDictDescriptor(BaseDataDescriptor[T], ABC):
    def store(self, data: T) -> dict[str, ValueType]:
        return {
            'value': str(data)
        }


class IntDescriptor(InDictDescriptor[int]):

    def load(self, dic: Dict[str, ValueType]) -> int:
        return int(dic['value'])  # type: ignore

    @classmethod
    def get_data_type(cls) -> Type[int]:
        return int


class FloatDescriptor(InDictDescriptor[float]):

    def load(self, dic: Dict[str, ValueType]) -> float:
        return float(dic['value'])  # type: ignore

    @classmethod
    def get_data_type(cls) -> Type[float]:
        return float


class StrDescriptor(InDictDescriptor[str]):

    def load(self, dic: Dict[str, ValueType]) -> str:
        return dic['value']  # type: ignore

    @classmethod
    def get_data_type(cls) -> Type[str]:
        return str

class ListDescriptor(BaseDataDescriptor[List[T]]):
    def store(self, data: List[T]) -> Dict[str, ValueType]:
        return {
            "list": data
        }

    def load(self, dic: Dict[str, ValueType]) -> List[T]:
        return cast(List[T], dic["list"])

    @classmethod
    def get_data_type(cls) -> Type[list]:
        return list


class BytesDescriptor(BaseDataDescriptor[bytes]):
    def store(self, data: bytes) -> Dict[str, ValueType]:
        filename = f"bytes-{self.block_name}-{self.get_timestamp_str()}.dat"
        filename = os.path.abspath(os.path.join(self.artifacts_folder, filename))
        with open(filename, "wb") as file:
            file.write(data)
            return {
                "filename": filename
            }

    def load(self, dic: Dict[str, ValueType]) -> bytes:
        filename = cast(str, dic['filename'])
        with open(filename, "rb") as file:
            data = file.read()
            return data

    @classmethod
    def get_data_type(cls) -> Type[bytes]:
        return bytes

class DictDescriptor(BaseDataDescriptor[dict]):
    def store(self, data: dict) -> dict[str, ValueType]:
        return data

    def load(self, dic: dict[str, ValueType]) -> dict:
        return dic

    def get_data_type(self) -> type[dict]:
        return dict