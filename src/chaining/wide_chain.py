from __future__ import annotations

from typing import List, Set, Dict

import numpy as np
from numpy import typing as npt

from src.chaining import Chain


class WideChain(Chain):

    def __init__(
        self,
        source: str,
        target_begin_pos: int,
        source_begin_pos: int,
        likelihoods: List[float] | npt.NDArray[np.float32] | None = None,
    ):
        super().__init__(source, target_begin_pos, source_begin_pos, None, "")
        self.likelihoods = np.array([] if likelihoods is None else likelihoods, dtype=np.float32)

    # ----- #
    # Magic #
    # ----- #

    def __len__(self) -> int:
        return len(self.likelihoods)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Chain):
            return NotImplemented

        if not isinstance(other, WideChain):
            return False

        return (
            self.source == other.source and self.source_begin_pos == other.source_begin_pos and len(self) == len(other)
        )

    def __hash__(self):
        return hash((self.source, self.source_begin_pos, len(self)))

    def __add__(self, other) -> Chain:
        return NotImplemented

    def __str__(self) -> str:
        return (
            "WideChain(\n"
            f"\ttarget: [{self.target_begin_pos};{self.target_end_pos}) ~ {self.get_score()}\n"
            f'\tsource: [{self.source_begin_pos};{self.source_end_pos}) ~ "{self.source}"\n'
            f"\tlen: {len(self)}"
            ")"
        )

    def __repr__(self) -> str:
        return (
            f"WideChain("
            f"source={self.source!r}, "
            f"target_begin_pos={self.target_begin_pos}, "
            f"source_begin_pos={self.source_begin_pos,}, "
            f"likelihoods={self.likelihoods!r}"
            f")"
        )

    def target_len(self) -> int:
        return len(self)

    def source_len(self) -> int:
        return len(self)

    @property
    def target_end_pos(self) -> int:
        return self.target_begin_pos + len(self)

    @property
    def source_end_pos(self) -> int:
        return self.source_begin_pos + len(self)

    def source_matches(self) -> Dict[int, int | None]:
        return {i + self.target_begin_pos: i + self.source_begin_pos for i in range(len(self))}

    @property
    def target_begin_skips(self) -> int:
        return 0

    @property
    def target_end_skips(self) -> int:
        return 0

    @property
    def source_begin_skips(self) -> int:
        return 0

    @property
    def source_end_skips(self) -> int:
        return 0

    def get_target_likelihood(self, target_pos: int) -> float:
        return self.likelihoods[target_pos - self.target_begin_pos].item()

    def get_score(self) -> float:
        return np.exp(np.log(self.likelihoods).sum()).item()

    def significant_len(self) -> int:
        raise NotImplementedError("This method is not implemented")

    def to_dict(self) -> dict:
        pass

    @classmethod
    def from_dict(cls, d: dict) -> "Chain":
        pass

    @classmethod
    def generate_chains(
        cls,
        source_likelihoods: npt.NDArray[np.float32],
        source_name: str,
        target_token_ids: List[int],
        target_start_pos: int,
    ) -> List[Chain]:
        raise NotImplementedError("Not supported, call generate_chains_bidirectionally instead")

    @classmethod
    def generate_chains_bidirectionally(
        cls,
        source_likelihoods: npt.NDArray[np.float32],
        source_name: str,
        target_token_ids: List[int],
        target_start_pos: int,
    ) -> List[Chain]:

        result_chains = []
        ids_slice = target_token_ids[:]
        for s_pos in range(len(source_likelihoods)):
            likelihoods_slice = source_likelihoods[s_pos:s_pos + len(target_token_ids)]
            # TODO: Implement








