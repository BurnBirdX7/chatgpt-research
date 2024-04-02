from __future__ import annotations

import copy

import numpy as np
import numpy.typing as npt
import torch
from typing import Optional, List, Set


class Chain:
    likelihood_significance_threshold: float = 1e-4
    max_consecutive_skips: int = 3

    def __init__(self, source: str,
                 begin_pos: int = None,
                 likelihoods: Optional[npt.NDArray[np.float64]] = None,
                 skip_count: int = 0,
                 parent_chain: Optional[Chain] = None):
        self.begin_pos: int = begin_pos     # inclusive
        self.likelihoods: npt.NDArray[np.float64] = [] if (likelihoods is None) else likelihoods
        self.source: str = source
        self.skip_count = skip_count
        self.parent_chain: Optional[Chain] = parent_chain

    @classmethod
    def new_empty(cls, source_name: str, pos: int) -> Chain:
        return cls(source_name, pos, np.zeros(0, dtype=np.float64), 0)

    @property
    def significant_likelihoods(self) -> List[float]:
        return list(filter(lambda x: x > Chain.likelihood_significance_threshold,
                           self.likelihoods))

    @property
    def end_pos(self) -> int:
        return self.begin_pos + len(self.likelihoods)

    def __len__(self) -> int:
        if self.begin_pos is None:
            return 0

        return self.end_pos - self.begin_pos

    def __str__(self) -> str:
        if self.parent_chain is None:
            return f"Chain({self.begin_pos}..{self.end_pos}  '{self.source}' ~ {self.get_score()})"
        return (f"Chain(\n"
                f"\t{self.begin_pos}..{self.end_pos}  '{self.source}' ~ {self.get_score()}\n"
                f"\tparent: {self.parent_chain!s}\n"
                f") ")

    def __repr__(self) -> str:
        return (f"Chain("
                f"begin_pos={self.begin_pos}, "
                f"end_pos={self.end_pos}, "
                f"likelihoods={self.likelihoods!r}, "
                f"source={self.source!r}, "
                f"skip_count={self.skip_count}, "
                f"parent_chain={self.parent_chain!r}"
                f")")

    def reverse(self) -> Chain:
        """
        Reverses order of likelihoods and makes 'begin' position of the chain its 'end' position
        """
        begin_pos = self.begin_pos - len(self.likelihoods)
        new_chain = Chain(self.source, begin_pos, np.flip(self.likelihoods), self.skip_count)

        print(f"Was: {self}")
        print(f"Was: {new_chain}")

        return new_chain

    def to_dict(self) -> dict:
        return {
            "begin_pos": self.begin_pos,
            "likelihoods": self.likelihoods,
            "skips": self.skip_count,
            "source": self.source
        }

    @staticmethod
    def from_dict(d: dict) -> "Chain":
        return Chain(
            begin_pos=d["begin_pos"],
            likelihoods=d["likelihoods"],
            skip_count=d["skips"],
            source=d["source"]
        )

    def append(self, likelihood: float) -> None:
        self.likelihoods = np.append(self.likelihoods, likelihood)

    def pop(self, n: int = 1) -> None:
        self.likelihoods = self.likelihoods[:-n]

    def get_token_positions(self) -> Set[int]:
        return set(range(self.begin_pos, self.end_pos))

    def get_score(self):
        # log2(2 + len) * ((lik_h_0 * ... * lik_h_len) ^ 1 / len)   = score
        likelihoods = self.significant_likelihoods
        l = np.exp(np.log(likelihoods).mean())
        score = l * (len(likelihoods)**2)
        return score

    def get_all_subchains(self) -> List[Chain]:
        parent = self if self.parent_chain is None else self.parent_chain

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

    def __add__(self, other: Chain) -> Chain:
        if not isinstance(other, Chain):
            return NotImplemented

        if self.source != other.source:
            raise ValueError(f"Cannot add two chains from different sources: {self.source} != {other.source}")

        if self.end_pos != other.begin_pos:
            raise ValueError(f"Can add only chains with beginning and the end on the same position, "
                             f"got chains: \n{self}\nand\n{other}")

        return Chain(self.source,
                     self.begin_pos,
                     np.concatenate([self.likelihoods, other.likelihoods]),
                     self.skip_count + other.skip_count)

    @staticmethod
    def __expand_chain(source_likelihoods: torch.Tensor, source_start_pos: int, source_name: str,
                       target_token_ids: List[int], target_start_pos: int) -> Chain | None:
        source_len = len(source_likelihoods)
        target_len = len(target_token_ids)

        # TODO: Get rid of the function and make expansion in 2 loops

        def expand_one_way(shift_lower_bound: int, shift_upper_bound: int, shift_step: int) -> Chain:
            chain = Chain.new_empty(source_name, source_start_pos)
            skips = 0
            for shift in range(shift_lower_bound, shift_upper_bound, shift_step):
                token_pos = target_start_pos + shift
                source_pos = source_start_pos + shift

                curr_id = target_token_ids[token_pos]
                curr_likelihood = source_likelihoods[source_pos][curr_id].item()

                chain.append(curr_likelihood)

                if curr_likelihood < Chain.likelihood_significance_threshold:
                    chain.skip_count += 1
                    skips += 1
                    if skips > Chain.max_consecutive_skips:
                        break
                else:
                    skips = 0

            chain.pop(skips)  # Remove skipped tokens that are placed on the end of the chain
            return chain
        # end: expand_one_way

        # backward_shift_upper_bound = max(-source_start_pos, -target_start_pos)
        forward_shift_upper_bound = min(source_len - source_start_pos, target_len - target_start_pos)

        # backward_chain = expand_one_way(-1, backward_shift_upper_bound, -1).reverse()
        forward_chain = expand_one_way(0, forward_shift_upper_bound, +1)

        return forward_chain

    @staticmethod
    def generate_chains(source_likelihoods: torch.Tensor, source_name: str,
                        target_token_ids: List[int], target_start_pos: int) -> List[Chain]:
        """
        Generates chains of tokens with the same source

        :param source_likelihoods: inferred from the source text likelihoods for the tokens
        :param source_name: name of the source, doesn't affect generation, all produced chains will have this name
        :param target_token_ids: token ids of the target text
        :param target_start_pos: position from where to start building chains
        """
        result_chains: List[Chain] = []

        source_len = len(source_likelihoods)
        for source_start_pos in range(0, source_len):
            chain = Chain.__expand_chain(source_likelihoods, source_start_pos, source_name,
                                         target_token_ids, target_start_pos)
            result_chains += chain.get_all_subchains()

        return result_chains
