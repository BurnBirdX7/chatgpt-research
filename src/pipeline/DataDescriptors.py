from __future__ import annotations

import os.path
from abc import ABC
from typing import TypeVar, cast

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

    def load(self, dic: dict[str, str]) -> int:
        return int(dic['value'])

    @classmethod
    def get_data_type(cls) -> type[int]:
        return int


class FloatDescriptor(InDictDescriptor[float]):

    def load(self, dic: dict[str, str]) -> float:
        return float(dic['value'])

    @classmethod
    def get_data_type(cls) -> type[float]:
        return float


class StrDescriptor(InDictDescriptor[str]):

    def load(self, dic: dict[str, str]) -> str:
        return dic['value']

    @classmethod
    def get_data_type(cls) -> type[str]:
        return str

class ListDescriptor(BaseDataDescriptor[list[T]]):
    def store(self, data: list[T]) -> dict[str, Value]:
        return {
            "list": data
        }

    def load(self, dic: dict[str, Value]) -> list[T]:
        return cast(T, dict["list"])

    @classmethod
    def get_data_type(cls) -> type[list]:
        return list


class BytesDescriptor(BaseDataDescriptor[bytes]):
    def store(self, data: bytes) -> dict[str, str]:
        filename = f"bytes-{self.block_name}-{self.get_timestamp_str()}.dat"
        filename = os.path.abspath(os.path.join(self.artifacts_folder, filename))
        with open(filename, "wb") as file:
            file.write(data)
            return {
                "filename": filename
            }

    def load(self, dic: dict[str, str]) -> bytes:
        filename = dic['filename']
        with open(filename, "rb") as file:
            data = file.read()
            return data

    @classmethod
    def get_data_type(cls) -> type[bytes]:
        return bytes
