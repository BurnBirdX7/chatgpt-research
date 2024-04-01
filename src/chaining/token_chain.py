from __future__ import annotations

import copy

import numpy as np
import torch
from typing import Optional, List, Set


class Chain:
    begin_pos: int
    end_pos: int
    likelihoods: List[float]
    skips: int = 0
    source: str

    def __init__(self, source: str,
                 begin_pos: int = None,   # type: ignore
                 end_pos: int = None,     # type: ignore
                 likelihoods: Optional[List[float]] = None,
                 skips: int = 0):
        self.begin_pos: int = begin_pos
        self.end_pos: int = end_pos
        self.likelihoods = [] if (likelihoods is None) else likelihoods
        self.source = source
        self.skips = skips

    def __len__(self) -> int:
        if self.begin_pos is None:
            return 0

        return self.end_pos - self.begin_pos + 1

    def __str__(self) -> str:
        return (f"Chain {{\n"
                f"\tseq = {self.begin_pos}..{self.end_pos}\n"
                f"\tlikelihoods = {self.likelihoods}\n"
                f"\tskips = {self.skips}\n"
                f"\tscore = {self.get_score()}\n"
                f"\tsource = {self.source}\n"
                f"}}\n")

    def __repr__(self) -> str:
        return (f"Chain("
                f"begin_pos={self.begin_pos}, "
                f"end_pos={self.end_pos}, "
                f"likelihoods={self.likelihoods!r}, "
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
            end_pos=d["end_pos"],
            likelihoods=d["likelihoods"],
            skips=d["skips"],
            source=d["source"]
        )

    def append(self, likelihood: float, position: int) -> None:
        self.likelihoods.append(likelihood)
        if self.begin_pos is None:
            self.begin_pos = position
            self.end_pos = position
        else:
            if self.end_pos + self.skips + 1 != position:
                raise ValueError(f"{self.end_pos=}, {position=}")
            self.end_pos += self.skips + 1
        self.skips = 0

    def skip(self) -> None:
        self.skips += 1

    def get_token_positions(self) -> Set[int]:
        return set(range(self.begin_pos, self.end_pos + 1))

    def get_score(self):
        # log2(2 + len) * ((lik_h_0 * ... * lik_h_len) ^ 1 / len)   = score
        l = np.exp(np.log(self.likelihoods).mean())
        score = l * (len(self.likelihoods)**2)
        return score

    @staticmethod
    def generate_chains(source_len: int, likelihoods: torch.Tensor,
                        token_ids: List[int], token_start_pos: int, source_name: str) -> List[Chain]:
        """
        Generates chains of tokens with the same source

        :param source_len: length of the source
        :param likelihoods: inferred from the source text likelihoods for the tokens
        :param token_ids: token ids of the target text
        :param token_start_pos: position from where to start building chains
        :param source_name: name of the source, doesn't affect generation, all produced chains will have this name
        """
        result_chains: List[Chain] = []

        for source_start_pos in range(0, source_len):
            chain = Chain(source_name)
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
                    if len(chain) > 1:
                        result_chains.append(copy.deepcopy(chain))

        return result_chains
