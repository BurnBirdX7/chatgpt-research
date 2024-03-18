from __future__ import annotations

import os.path
from abc import ABC
from typing import TypeVar, cast, List, Dict, Type

from .BaseDataDescriptor import BaseDataDescriptor, Value

T = TypeVar('T')

class EmptyDataDescriptor(BaseDataDescriptor[None]):
    @classmethod
    def get_data_type(cls) -> type[None]:
        return type(None)

    def store(self, data: None) -> dict[str, str]:
        return {}

    def load(self, dic: dict[str, str]) -> None:
        return None

    def is_type_compatible(self, typ: type | None):
        return typ is None or issubclass(typ, type(None))


class InDictDescriptor(BaseDataDescriptor[T], ABC):
    def store(self, data: T) -> dict[str, str]:
        return {
            'value': str(data)
        }


class IntDescriptor(InDictDescriptor[int]):

    def load(self, dic: Dict[str, str]) -> int:
        return int(dic['value'])

    @classmethod
    def get_data_type(cls) -> Type[int]:
        return int


class FloatDescriptor(InDictDescriptor[float]):

    def load(self, dic: Dict[str, str]) -> float:
        return float(dic['value'])

    @classmethod
    def get_data_type(cls) -> Type[float]:
        return float


class StrDescriptor(InDictDescriptor[str]):

    def load(self, dic: Dict[str, str]) -> str:
        return dic['value']

    @classmethod
    def get_data_type(cls) -> Type[str]:
        return str

class ListDescriptor(BaseDataDescriptor[List[T]]):
    def store(self, data: List[T]) -> Dict[str, Value]:
        return {
            "list": data
        }

    def load(self, dic: Dict[str, Value]) -> List[T]:
        return cast(T, Dict["list"])

    @classmethod
    def get_data_type(cls) -> Type[list]:
        return list


class BytesDescriptor(BaseDataDescriptor[bytes]):
    def store(self, data: bytes) -> Dict[str, str]:
        filename = f"bytes-{self.block_name}-{self.get_timestamp_str()}.dat"
        filename = os.path.abspath(os.path.join(self.artifacts_folder, filename))
        with open(filename, "wb") as file:
            file.write(data)
            return {
                "filename": filename
            }

    def load(self, dic: Dict[str, str]) -> bytes:
        filename = dic['filename']
        with open(filename, "rb") as file:
            data = file.read()
            return data

    @classmethod
    def get_data_type(cls) -> Type[bytes]:
        return bytes
