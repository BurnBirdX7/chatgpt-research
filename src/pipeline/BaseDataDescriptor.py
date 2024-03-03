from __future__ import annotations

import datetime
import random
from typing import Generic, TypeVar
from abc import abstractmethod, ABC

T = TypeVar('T')


class BaseDataDescriptor(Generic[T], ABC):
    def __init__(self):
        self.artifacts_folder = "pipe-artifacts"
        self.block_name = "unnamed"

    def is_type_compatible(self, typ: type | None):
        if typ is None:
            return False
        return issubclass(typ, self.get_data_type())

    @abstractmethod
    def store(self, data: T) -> dict[str, str]:
        """
        Method stores data to the disk,
        and returns information crucial for its restoration as a dict
        """
        raise NotImplemented

    @abstractmethod
    def load(self, dic: dict[str, str]) -> T:
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
    def get_random_string() -> str:
        return str(random.randint(111_111, 999_999))
