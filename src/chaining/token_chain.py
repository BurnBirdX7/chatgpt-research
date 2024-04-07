from __future__ import annotations

import copy
import itertools

import numpy as np
import numpy.typing as npt
import torch
from typing import Optional, List, Set


class Chain:
    likelihood_significance_threshold = 1e-5
    max_skips = 3

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

    def __add__(self, other: Chain) -> Chain:
        if not isinstance(other, Chain):
            return NotImplemented

        if self.source != other.source:
            raise ValueError(f"Added chain must have same source, got: {self.source} and {other.source}")

        if self.end_pos - 1 != other.begin_pos:
            raise ValueError("Added chains must have one common position"
                             "on the end of left chain and beginning of the right chain. "
                             f"Got chains:\n{self}\nand\n{other}")

        if self.all_likelihoods[-1] != other.all_likelihoods[0]:
            raise ValueError("Added chains must have one common likelihood"
                             "on the end of left chain and beginning of the right chain. "
                             f"Got chains:\n{self}\nand\n{other}"
                             f"with likelihoods: {other.all_likelihoods[-1]} and {other.all_likelihoods[0]}")

        chain = Chain(
            source=self.source,
            begin_pos=self.begin_pos,
            all_likelihoods=np.concatenate([self.all_likelihoods[:-1], other.all_likelihoods])
        )

        assert len(chain) == (len(self) + len(other) - 1)

        return chain

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

    def reverse(self) -> Chain:
        rev_chain = Chain(
            source=self.source,
            begin_pos=self.begin_pos - len(self) + 1,
            all_likelihoods=np.flip(self.all_likelihoods),
            parent=self
        )

        rev_chain._begin_skips, rev_chain._end_skips = self._end_skips, self._begin_skips

        assert rev_chain.end_pos == self.begin_pos + 1
        return rev_chain

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
            # FORWARD CHAINING:
            chain = Chain(source_name, target_start_pos)
            skips = 0
            shift_upper_bound = min(source_len - source_start_pos, len(target_token_ids) - target_start_pos)
            for shift in range(0, shift_upper_bound):
                target_pos = target_start_pos + shift
                source_pos = source_start_pos + shift

                assert target_pos < len(target_token_ids)
                assert source_pos < source_len

                token_curr_id = target_token_ids[target_pos]
                token_curr_likelihood = source_likelihoods[source_pos][token_curr_id].item()

                if token_curr_likelihood < Chain.likelihood_significance_threshold:
                    chain.skip_end(token_curr_likelihood)
                    skips += 1
                    if skips > Chain.max_skips:
                        break
                else:
                    chain.append_end(token_curr_likelihood)
                    if len(chain) > 1:
                        result_chains.append(copy.copy(chain))

        return result_chains

    @staticmethod
    def generate_chains_bidirectional(source_likelihoods: torch.Tensor, source_name: str,
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
            # FORWARD CHAINING:
            forward_chain = Chain(source_name, target_start_pos)
            forward_chains = []
            skips = 0
            shift_upper_bound = min(source_len - source_start_pos, len(target_token_ids) - target_start_pos)
            for shift in range(0, shift_upper_bound):
                target_pos = target_start_pos + shift
                source_pos = source_start_pos + shift

                assert target_pos < len(target_token_ids)
                assert source_pos < source_len

                token_curr_id = target_token_ids[target_pos]
                token_curr_likelihood = source_likelihoods[source_pos][token_curr_id].item()

                if token_curr_likelihood < Chain.likelihood_significance_threshold:
                    forward_chain.skip_end(token_curr_likelihood)
                    skips += 1
                    if skips > Chain.max_skips:
                        break
                else:
                    forward_chain.append_end(token_curr_likelihood)
                    if len(forward_chain) > 1:
                        forward_chains.append(copy.copy(forward_chain))

            # BACKWARD CHAINING:
            backward_chain = Chain(source_name, target_start_pos)
            backward_chains = []
            skips = 0
            shift_upper_bound = min(source_start_pos, target_start_pos)
            for shift in range(0, shift_upper_bound):
                target_pos = target_start_pos - shift
                source_pos = source_start_pos - shift

                assert target_pos >= 0
                assert source_pos >= 0

                token_curr_id = target_token_ids[target_pos]
                token_curr_likelihood = source_likelihoods[source_pos][token_curr_id].item()

                if token_curr_likelihood < Chain.likelihood_significance_threshold:
                    backward_chain.skip_end(token_curr_likelihood)
                    skips += 1
                    if skips > Chain.max_skips:
                        break
                else:
                    backward_chain.append_end(token_curr_likelihood)
                    if len(backward_chain) > 0:
                        backward_chains.append(backward_chain.reverse())

            assert backward_chain.begin_pos == forward_chain.begin_pos

            # COMBINE CHAINS:
            for backward_chain, forward_chain in itertools.product(backward_chains, forward_chains):
                if backward_chain._end_skips + forward_chain._begin_skips > Chain.max_skips:
                    # Reject chain combinations with a large gap in the middle
                    continue

                result_chains.append(backward_chain + forward_chain)

            # Add uni-directional chains to the set
            for chain in backward_chains:
                if chain._end_skips > 0:
                    chain.trim()

            for chain in forward_chains:
                if chain._begin_skips > 0:
                    chain.trim()

            result_chains += backward_chains + forward_chains

        return result_chains
