from __future__ import annotations

import copy
import logging

import numpy as np
import numpy.typing as npt
import torch
from typing import Optional, List, Set


class Chain:

    def __init__(self, source: str,
                 begin_pos: int,
                 likelihoods: Optional[List[float] | npt.NDArray[np.float64]] = None,
                 all_likelihoods: Optional[List[float] | npt.NDArray[np.float64]] = None,
                 skips: int = 0):
        self.begin_pos: int = begin_pos
        self.likelihoods = np.array([] if (likelihoods is None) else likelihoods)
        self.all_likelihoods = np.array([] if (all_likelihoods is None) else all_likelihoods)
        self.source = source
        self.skips = skips

    @property
    def end_pos(self) -> int:
        return self.begin_pos + len(self)

    def __len__(self) -> int:
        return len(self.all_likelihoods)

    def __str__(self) -> str:
        return f"Chain({self.begin_pos}..{self.end_pos}  '{self.source}' ~ {self.get_score()})"

    def __repr__(self) -> str:
        return (f"Chain("
                f"pos={self.begin_pos}, "
                f"likelihoods={self.likelihoods!r}, "
                f"all_likelihoods={self.all_likelihoods!r}"
                f"source={self.source!r}, "
                f"skips={self.skips}"
                f")")

    def to_dict(self) -> dict:
        return {
            "begin_pos": self.begin_pos,
            "end_pos": self.end_pos,
            "likelihoods": self.likelihoods,
            "skips": self.skips,
            "source": self.source
        }

    @staticmethod
    def from_dict(d: dict) -> "Chain":
        return Chain(
            begin_pos=d["begin_pos"],
            likelihoods=d["likelihoods"],
            skips=d["skips"],
            source=d["source"]
        )

    def append(self, likelihood: float, position: int) -> None:
        self.all_likelihoods = np.append(self.all_likelihoods, likelihood)
        self.likelihoods = np.append(self.likelihoods, likelihood)
        self.skips = 0

    def skip(self) -> None:
        self.all_likelihoods = np.append(self.all_likelihoods, 0.0)
        self.skips += 1

    def trim(self):
        insignificant = self.all_likelihoods < 1e-5
        trim_front = 0
        while trim_front < len(insignificant) and insignificant[trim_front]:
            trim_front += 1

        self.all_likelihoods = self.all_likelihoods[trim_front:]
        self.begin_pos += trim_front

        trim_back = len(insignificant) - 1
        while trim_front >= 0 and insignificant[trim_back]:
            trim_back -= 1

        self.all_likelihoods = self.all_likelihoods[:trim_back+1]


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
            shift_upper_bound = min(source_len - source_start_pos, len(token_ids) - token_start_pos)
            for shift in range(0, shift_upper_bound):
                token_pos = token_start_pos + shift
                source_pos = source_start_pos + shift

                assert token_pos < len(token_ids)
                assert source_pos < source_len

                token_curr_id = token_ids[token_pos]
                token_curr_likelihood = likelihoods[source_pos][token_curr_id].item()

                if token_curr_likelihood < 1e-5:
                    chain.skip()
                    if chain.skips > 3:
                        break
                else:
                    chain.append(token_curr_likelihood, token_pos)
                    new_chain = copy.deepcopy(chain)
                    new_chain.trim()
                    if len(new_chain) > 1:
                        # Experiment and report
                        result_chains.append(copy.deepcopy(new_chain))

        return result_chains
