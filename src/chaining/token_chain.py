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
                 likelihoods: Optional[List[float] | npt.NDArray[np.float64]] = None,
                 all_likelihoods: Optional[List[float] | npt.NDArray[np.float64]] = None,
                 parent: Optional[Chain] = None):
        self.begin_pos: int = begin_pos
        self.likelihoods = np.array([] if (likelihoods is None) else likelihoods)
        self.all_likelihoods = np.array([] if (all_likelihoods is None) else all_likelihoods)
        self.source = source
        self.skips_count = len(self.all_likelihoods) - len(self.likelihoods)
        self.parent = parent

        self._begin_skips = 0
        self._end_skips = 0


    @property
    def end_pos(self) -> int:
        return self.begin_pos + len(self)

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
            likelihoods=d["likelihoods"],
            all_likelihoods=d["all_likelihoods"],
            source=d["source"]
        )

    def append_end(self, likelihood: float) -> None:
        self.all_likelihoods = np.append(self.all_likelihoods, likelihood)
        self.likelihoods = np.append(self.likelihoods, likelihood)
        self._end_skips = 0

    def skip_end(self, likelihood: float) -> None:
        self.all_likelihoods = np.append(self.all_likelihoods, likelihood)
        self.skips_count += 1
        self._end_skips += 1
        if len(self) == 0:
            self._begin_skips += 1

    def trim(self):
        end_trim = len(self) - self._end_skips
        self.all_likelihoods = self.all_likelihoods[self._begin_skips:end_trim]
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

    def get_all_subchains(self) -> List[Chain]:
        parent = self if self.parent is None else self.parent

        pos = self.begin_pos
        skip_mask: npt.NDArray[np.bool_] = self.likelihoods < Chain.likelihood_significance_threshold

        n = len(self)
        subchains = [
            Chain(
                self.source,
                pos + i,
                self.likelihoods[i:i+l],
                skip_mask[i:i+l].sum(),
                parent
            )
            for l in range(2, n)
            for i in range(0, n - l)
        ]

        print(f"Subchains count: {len(subchains)}, {self}")
        return subchains

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
                    new_chain = chain.trim_copy()
                    if len(new_chain) > 1:
                        result_chains.append(new_chain)

        return result_chains
