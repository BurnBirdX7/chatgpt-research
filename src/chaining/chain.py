from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np
import numpy.typing as npt


class Chain(ABC):

    def __init__(
        self,
        source: str,
        target_begin_pos: int,
        source_begin_pos: int,

        parent: Chain | List[Chain] | None,
        cause: str
    ):
        self.source = source
        self.target_begin_pos = target_begin_pos
        self.source_begin_pos = source_begin_pos

        self.parent = parent
        self.cause = cause

    # ----- #
    # Magic #
    # ----- #

    @abstractmethod
    def __eq__(self, other) -> bool | NotImplemented: ...

    @abstractmethod
    def __add__(self, other) -> Chain: ...

    @abstractmethod
    def __hash__(self): ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __str__(self) -> str: ...

    @abstractmethod
    def __repr__(self) -> str: ...

    # --------- #
    # Positions #
    # --------- #

    @property
    @abstractmethod
    def target_end_pos(self) -> int: ...

    @property
    @abstractmethod
    def source_end_pos(self) -> int: ...

    # ---------------- #
    # Skip information #
    # ---------------- #

    @property
    @abstractmethod
    def target_end_skips(self) -> int: ...

    @property
    @abstractmethod
    def target_begin_skips(self) -> int: ...

    @property
    @abstractmethod
    def source_end_skips(self) -> int:
        return ...

    @property
    @abstractmethod
    def source_begin_skips(self) -> int: ...

    # ----------- #
    # likelihoods #
    # ----------- #

    @property
    @abstractmethod
    def significant_likelihoods(self) -> npt.NDArray[np.float32]: ...

    @abstractmethod
    def get_target_likelihood(self, target_pos: int) -> float: ...

    # ------------- #
    # Serialization #
    # ------------- #

    @abstractmethod
    def to_dict(self) -> dict: ...

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict) -> "Chain": ...

    # ------- #
    # Parents #
    # ------- #

    def parent_str(self) -> str:
        if self.parent is None:
            return "-"

        if isinstance(self.parent, Chain):
            return str(self.parent).replace("\n\t", "\n\t\t")

        return "[" + ", \n\t".join([str(chain) for chain in self.parent]).replace("\n\t", "\n\t\t") + "]"
