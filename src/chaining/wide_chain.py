from __future__ import annotations

from typing import List, Dict, Sequence

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

        return (self.source == other.source
                and self.source_begin_pos == other.source_begin_pos
                and len(self) == len(other))

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
        return {
            'source': self.source,
            'target_begin_pos': self.target_begin_pos,
            'source_begin_pos': self.source_begin_pos,
            'likelihoods': self.likelihoods.tolist()
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chain":
        return WideChain(
            source=d['source'],
            target_begin_pos=d['target_begin_pos'],
            source_begin_pos=d['source_begin_pos'],
            likelihoods=d['likelihoods']
        )

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
    ) -> Sequence[Chain]:

        # target_start_pos is ignored

        # [doc]
        # ls = len(source), lt = len(target)

        # Long source, short target
        #
        # [.target..] ------> slides ------> [.target..]
        #         [.source.....................]
        #
        # n-th step, n <= lt          | n > lt                         | n > ls
        # target_window = target[-n:] | target_window = target[:]      | target_window = target[:-n+ls]
        # source_window = source[:n]  | source_window = source[n-lt:n] | source_window = source[n-ls:]

        # Short source, long target, >sliding target<
        #
        # [.target.........................] --> [.target.........................]
        #                                [.source..]
        #
        # n-th step, n <= ls          | n > ls                               | n > lt
        # target_window = target[-n:] | target_window = target[lt-n:lt-n+ls] | target_window = target[:-n+lt]
        #           ... = ...         |           ... = target[n-ls:n]       |           ... = ...
        # source_window = source[:n]  | source_window = source[:]            | source_window = source[n-lt:]

        lt, ls = len(target_token_ids), len(source_likelihoods)
        bound_1 = min(lt, ls)
        bound_2 = max(lt, ls)

        result_chains = []
        target_token_ids = np.array(target_token_ids)

        def create_chain(target_window: npt.NDArray[np.int64],
                         source_window: npt.NDArray[np.float32],
                         target_begin_pos: int,
                         source_begin_pos: int):

            likelihoods = [
                vec[idx]
                for idx, vec in zip(target_window, source_window)
            ]
            result_chains.append(WideChain(source_name, target_begin_pos, source_begin_pos, likelihoods))

        for n in range(1, bound_1 + 1):
            target_window = target_token_ids[-n:]
            source_window = source_likelihoods[:n]
            create_chain(target_window, source_window, lt - n, 0)

        if ls > lt:
            for n in range(bound_1 + 1, bound_2 + 1):
                target_window = target_token_ids[:]
                source_window = source_likelihoods[n - bound_1:n]
                create_chain(target_window, source_window, 0, n - bound_1)
        else:
            for n in range(bound_1 + 1, bound_2 + 1):
                target_window = target_token_ids[n - bound_1:n]
                source_window = source_likelihoods[:]
                create_chain(target_window, source_window, n - bound_1, 0)

        for n in range(bound_2 + 1, lt + ls):
            target_window = target_token_ids[:-n + bound_2]
            source_window = source_likelihoods[n - bound_2:]
            create_chain(target_window, source_window, 0, n - bound_2)

        return result_chains
