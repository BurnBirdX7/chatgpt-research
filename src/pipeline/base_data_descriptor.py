from __future__ import annotations

__all__ = ['BaseDataDescriptor', 'ValueType']

import datetime
import pathlib
import random
import string
from typing import Generic, TypeVar, Union, List, Dict
from abc import abstractmethod, ABC

T = TypeVar('T')
ValueType = Union[int, float, str, List["ValueType"], Dict[str, "ValueType"]]


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
        """Method used to store data to the disk

        Parameters
        ----------
        data : T
            Data that should be stored on disk

        Returns
        -------
        dict[str, ValueType]
            Dictionary that contains arbitrary information that describes how to restore data.
            MUST be acceptable by overloaded ``load`` method

        Notes
        -----
        This method can and SHOULD (if possible) just convert data into the dictionary form and return it.

        [!] You can write data to the file in this method, but if you create new files in this method,
        report them in the returned dictionary and override ``cleanup`` method
        to be able to remove unneeded files on request.
        Check its default implementation.
        """

    @abstractmethod
    def load(self, dic: dict[str, ValueType]) -> T:
        """Restores data from disk and returns it.
        Overload this method to implement loading saved data from the disk

        Parameters
        ----------
        dic : dict[str, ValueType]
            dictionary produced by `store` method of the same descriptor

        Returns
        -------
        T
            Data restored from disk, data should be as close as possible to what was stored
        """

    @staticmethod
    def cleanup_files(*filepath_list: str):
        """Removes listed files, doesn't throw if files are missing
        """
        for filepath in filepath_list:
            pathlib.Path.unlink(pathlib.Path(filepath), missing_ok=True)

    def cleanup(self, dic: dict[str, ValueType]):
        """Removes artifacts produces by this descriptor from the disk.
        Overload this method if your descriptor places data on disks and not only in returned dictionary.
        Use `cleanup_files` if possible.

        Parameters
        ----------
        dic : dict[str, ValueType]
            dictionary produced by `store` method of the same descriptor

        See also
        --------
        ``data_descriptors.BytesNode`` : Node that stores byte-string to the disk.
            Visit to see how file removal is handled there
        """

    @abstractmethod
    def get_data_type(self) -> type[T]:
        """Returns type of the data.
        Used for typechecking pipeline links.
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
