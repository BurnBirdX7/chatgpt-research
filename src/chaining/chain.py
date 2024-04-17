from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Set, Any, Dict

import numpy as np
import numpy.typing as npt


class Chain(ABC):

    def __init__(
        self, source: str, target_begin_pos: int, source_begin_pos: int, parent: Chain | List[Chain] | None, cause: str
    ):
        self.source = source
        self.target_begin_pos = target_begin_pos
        self.source_begin_pos = source_begin_pos

        self.parent = parent
        self.cause = cause

        self.attachment: Dict[str, Any] = {}

    # ----- #
    # Magic #
    # ----- #
    @abstractmethod
    def __eq__(self, other) -> bool: ...

    @abstractmethod
    def __hash__(self): ...

    @abstractmethod
    def __add__(self, other) -> Chain: ...

    @abstractmethod
    def __str__(self) -> str: ...

    @abstractmethod
    def __repr__(self) -> str: ...

    # --------- #
    # Positions #
    # --------- #

    @abstractmethod
    def target_len(self) -> int: ...

    @abstractmethod
    def source_len(self) -> int: ...

    @property
    def target_end_pos(self) -> int:
        return self.target_begin_pos + self.target_len()

    @property
    def source_end_pos(self) -> int:
        return self.source_begin_pos + self.source_len()

    @abstractmethod
    def source_matches(self) -> Dict[int, int | None]: ...

    def get_target_token_positions(self) -> Set[int]:
        return set(range(self.target_begin_pos, self.target_end_pos))

    @abstractmethod
    def get_source_token_positions(self) -> Set[int]:
        return set(range(self.source_begin_pos, self.source_end_pos))

    # ---------------- #
    # Skip information #
    # ---------------- #

    @property
    @abstractmethod
    def target_begin_skips(self) -> int: ...

    @property
    @abstractmethod
    def target_end_skips(self) -> int: ...

    @property
    @abstractmethod
    def source_begin_skips(self) -> int: ...

    @property
    @abstractmethod
    def source_end_skips(self) -> int: ...

    # ----------- #
    # likelihoods #
    # ----------- #

    @abstractmethod
    def get_target_likelihood(self, target_pos: int) -> float: ...

    @abstractmethod
    def get_score(self) -> float: ...

    @abstractmethod
    def significant_len(self) -> int: ...

    # ------------- #
    # Serialization #
    # ------------- #

    @abstractmethod
    def to_dict(self) -> dict: ...

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict) -> "Chain": ...

    # ------------------ #
    # Chaining functions #
    # ------------------ #

    @classmethod
    @abstractmethod
    def generate_chains(
        cls,
        source_likelihoods: npt.NDArray[np.float32],
        source_name: str,
        target_token_ids: List[int],
        target_start_pos: int,
    ) -> List[Chain]: ...

    @classmethod
    @abstractmethod
    def generate_chains_bidirectionally(
        cls,
        source_likelihoods: npt.NDArray[np.float32],
        source_name: str,
        target_token_ids: List[int],
        target_start_pos: int,
    ) -> List[Chain]: ...

    # ------- #
    # Parents #
    # ------- #

    def parent_str(self) -> str:
        if self.parent is None:
            return "-"

        if isinstance(self.parent, Chain):
            return str(self.parent).replace("\n\t", "\n\t\t")

        return "[" + ", \n\t".join([str(chain) for chain in self.parent]).replace("\n\t", "\n\t\t") + "]"
