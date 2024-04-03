from __future__ import annotations

import copy
import logging

import numpy as np
import numpy.typing as npt
import torch
from typing import Optional, List, Set


class Chain:
    likelihood_significance_threshold = 1e-5

    def __init__(self, source: str,
                 begin_pos: int,
                 all_likelihoods: Optional[List[float] | npt.NDArray[np.float64]] = None,
                 parent: Optional[Chain] = None):
        self.begin_pos: int = begin_pos
        self.all_likelihoods = np.array([] if (all_likelihoods is None) else all_likelihoods)
        self.source = source
        self.parent = parent

        self._begin_skips = 0
        self._end_skips = 0

    @property
    def end_pos(self) -> int:
        return self.begin_pos + len(self)

    @property
    def likelihoods(self) -> npt.NDArray[np.float64]:
        return self.all_likelihoods[self.all_likelihoods >= Chain.likelihood_significance_threshold]

    def __eq__(self, other: Chain) -> bool:
        if not isinstance(other, Chain):
            return False

        return (self.source == other.source and
                self.begin_pos == other.begin_pos and
                np.array_equal(self.all_likelihoods, other.all_likelihoods))


    def __len__(self) -> int:
        return len(self.all_likelihoods)

    def __str__(self) -> str:
        if self.parent is None:
            return f"Chain({self.begin_pos}..{self.end_pos}  '{self.source}' ~ {self.get_score()})"
        else:
            return (f"Chain(\n"
                    f"\t{self.begin_pos}..{self.end_pos}  '{self.source}' ~ {self.get_score()}\n"
                    f"\tparent: {self.parent!s}"
                    f")")

    def __repr__(self) -> str:
        return (f"Chain("
                f"pos={self.begin_pos}, "
                f"likelihoods={self.likelihoods!r}, "
                f"all_likelihoods={self.all_likelihoods!r}"
                f"source={self.source!r}, "
                f"parent={self.parent!r}"
                f")")

    def to_dict(self) -> dict:
        return {
            "begin_pos": self.begin_pos,
            "likelihoods": self.likelihoods,
            "all_likelihoods": self.all_likelihoods,
            "source": self.source
        }

    @staticmethod
    def from_dict(d: dict) -> "Chain":
        return Chain(
            begin_pos=d["begin_pos"],
            all_likelihoods=d["all_likelihoods"],
            source=d["source"]
        )

    def append_end(self, likelihood: float) -> None:
        self.all_likelihoods = np.append(self.all_likelihoods, likelihood)
        self._end_skips = 0

    def skip_end(self, likelihood: float) -> None:
        self.all_likelihoods = np.append(self.all_likelihoods, likelihood)
        self._end_skips += 1
        if len(self) == 0:
            self._begin_skips += 1

    def trim(self):
        """
        Trims chain based on the meta information about skips
        """
        end_trim = len(self) - self._end_skips
        self.all_likelihoods = self.all_likelihoods[self._begin_skips:end_trim]
        self.begin_pos += self._begin_skips
        self._begin_skips = 0
        self._end_skips = 0

    def trim_copy(self) -> Chain:
        obj = copy.deepcopy(self)
        obj.trim()
        return obj

    def get_token_positions(self) -> Set[int]:
        return set(range(self.begin_pos, self.end_pos))

    def get_score(self):
        # log2(2 + len) * ((lik_h_0 * ... * lik_h_len) ^ 1 / len)   = score
        l = np.exp(np.log(self.likelihoods).mean())
        score = l * (len(self.likelihoods)**2)
        return score

    @staticmethod
    def generate_chains(likelihoods: torch.Tensor, source_name: str,
                        token_ids: List[int], token_start_pos: int) -> List[Chain]:
        """
        Generates chains of tokens with the same source

        :param source_len: length of the source
        :param likelihoods: inferred from the source text likelihoods for the tokens
        :param token_ids: token ids of the target text
        :param token_start_pos: position from where to start building chains
        :param source_name: name of the source, doesn't affect generation, all produced chains will have this name
        """
        result_chains: List[Chain] = []
        source_len = len(likelihoods)

        for source_start_pos in range(0, source_len):
            chain = Chain(source_name, token_start_pos)
            skips = 0
            shift_upper_bound = min(source_len - source_start_pos, len(token_ids) - token_start_pos)
            for shift in range(0, shift_upper_bound):
                token_pos = token_start_pos + shift
                source_pos = source_start_pos + shift

                assert token_pos < len(token_ids)
                assert source_pos < source_len

                token_curr_id = token_ids[token_pos]
                token_curr_likelihood = likelihoods[source_pos][token_curr_id].item()

                if token_curr_likelihood < 1e-5:
                    chain.skip_end(token_curr_likelihood)
                    skips += 1
                    if skips > 3:
                        break
                else:
                    chain.append_end(token_curr_likelihood)
                    if len(chain) > 1:
                        result_chains.append(copy.copy(chain))

        return result_chains
