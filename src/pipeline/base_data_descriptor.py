from __future__ import annotations

__all__ = ['BaseDataDescriptor', 'ValueType']

import datetime
import random
import string
from typing import Generic, TypeVar, Union, List, Dict
from abc import abstractmethod, ABC

T = TypeVar('T')
ValueType = Union[int, float, str, list, dict, List["ValueType"], Dict[str, "ValueType"]]

class BaseDataDescriptor(Generic[T], ABC):
    def __init__(self):
        self.artifacts_folder = "pipe-artifacts"
        self.block_name = "unnamed"

    def __repr__(self) -> str:
        return f"<Descriptor:{self.__class__.__name__}>"

    def is_type_compatible(self, typ: type | None):
        if typ is None:
            return False
        return issubclass(typ, self.get_data_type())

    def is_optional(self) -> bool:
        return False

    @abstractmethod
    def store(self, data: T) -> dict[str, ValueType]:
        """
        Method stores data to the disk,
        and returns information crucial for its restoration as a dict
        """
        raise NotImplemented

    @abstractmethod
    def load(self, dic: dict[str, ValueType]) -> T:
        """
        Restores data from disk and returns it
        """
        raise NotImplemented

    @abstractmethod
    def get_data_type(self) -> type[T]:
        """
        Returns type of the data
        """
        raise NotImplemented

    # Helpers
    @staticmethod
    def get_timestamp_str() -> str:
        return BaseDataDescriptor.format_time(datetime.datetime.now())

    @staticmethod
    def format_time(time: datetime.datetime) -> str:
        return time.strftime("%Y-%m-%d.%H-%M-%S")

    @staticmethod
    def get_random_string(n: int = 16) -> str:
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))
