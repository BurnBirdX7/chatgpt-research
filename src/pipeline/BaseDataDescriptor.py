from __future__ import annotations

import datetime
import os.path
from typing import Generic, TypeVar
from abc import abstractmethod, ABC


T = TypeVar('T')


class BaseDataDescriptor(Generic[T], ABC):

    def __init__(self: "BaseDataDescriptor", block_name: str, artifacts_path: str) -> None:
        """
        :param block_name: block name, e.g. "int2str-0"
        :param artifacts_path: relative path to artifacts folder, recommended to use when saving data to disk
        """
        self.block_name = block_name
        self.artifacts_path = os.path.join(os.path.curdir, artifacts_path)

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

    @classmethod
    @abstractmethod
    def get_data_type(cls) -> type[T]:
        """
        Returns type of the data
        """
        raise NotImplemented

    @staticmethod
    def get_timestamp_str() -> str:
        return BaseDataDescriptor.format_time(datetime.datetime.now())

    @staticmethod
    def format_time(time: datetime.datetime) -> str:
        return time.strftime("%Y-%m-%d.%H-%M-%S")
