from __future__ import annotations

import copy
import itertools

import numpy as np
import numpy.typing as npt
import torch
from typing import Optional, List, Set


class Chain:
    likelihood_significance_threshold = 1e-5
    max_skips = 2

    def __init__(
        self,
        source: str,
        target_begin_pos: int,
        source_begin_pos: int,
        all_likelihoods: Optional[List[float] | npt.NDArray[np.float64]] = None,
        parent: Optional[Chain] = None,
    ):
        """Create chain

        Parameters
        ----------
        source : str
            source associated with the chain, arbitrary string, generally URL

        target_begin_pos : int
            first token position of the chain in the target text

        source_begin_pos : int
            first token position of the chain in matched source text

        all_likelihoods : List[float] or npt.NDArray[float], optional
            All likelihoods of tokens from target text appearing in source text.
            If no value is supplied, length of 0 is assumed

        parent : Chain, optional
            Parent chain, intended for debug purposes.
            Generally is a chain that was reduced or transformed into this chain
        """

        self.target_begin_pos: int = target_begin_pos
        self.source_begin_pos: int = source_begin_pos
        self.all_likelihoods = np.array(
            [] if (all_likelihoods is None) else all_likelihoods
        )
        self.source = source
        self.parent = parent
        self.matched_source_text: str | None = None

        self._begin_skips = 0
        self._end_skips = 0

    @property
    def target_end_pos(self) -> int:
        """Last (exclusive) token position of the chain in the target text"""
        return self.target_begin_pos + len(self)

    @property
    def source_end_pos(self) -> int:
        """Last (exclusive) token position of the chain in matched source text"""
        return self.source_begin_pos + len(self)

    @property
    def likelihoods(self) -> npt.NDArray[np.float64]:
        """Significant likelihoods"""
        return self.all_likelihoods[
            self.all_likelihoods >= Chain.likelihood_significance_threshold
        ]

    def __eq__(self, other: Chain) -> bool:
        if not isinstance(other, Chain):
            return False

        return (
            self.source == other.source
            and self.target_begin_pos == other.target_begin_pos
            and self.source_begin_pos == other.source_begin_pos
            and np.array_equal(self.all_likelihoods, other.all_likelihoods)
        )

    def __len__(self) -> int:
        """Length of the chain.
        Amount of tokens covered by the chain in target and source texts
        """
        return len(self.all_likelihoods)

    def __str__(self) -> str:
        return (
            f"Chain(\n"
            f"\ttarget: [{self.target_begin_pos};{self.target_end_pos}) ~ {self.get_score()}\n"
            f'\tsource: [{self.source_begin_pos};{self.source_end_pos}) ~ "{self.source}"\n'
            f"\tlen: {len(self)}\n"
            f"\tparent: {self.parent!s}"
            f")"
        )

    def __repr__(self) -> str:
        return (
            f"Chain("
            f"target_begin_pos={self.target_begin_pos}, "
            f"source_begin_pos={self.source_begin_pos, }"
            f"likelihoods={self.likelihoods!r}, "
            f"all_likelihoods={self.all_likelihoods!r}"
            f"source={self.source!r}, "
            f"parent={self.parent!r}"
            f")"
        )

    def __add__(self, other: Chain) -> Chain:
        """Adds 2 chains.
        These chains must share the same token positions at the beginning and the end

        Notes
        -----
        Chains must overlap.
        That means that left chain's last token positions and likelihood must match
        first token positions and likelihood of the right chain

        Raises
        ------
        ValueError
            if chains have different sources or couldn't be added due to not being neighbours
        """
        if not isinstance(other, Chain):
            return NotImplemented

        if self.source != other.source:
            raise ValueError(
                f"Added chain must have same source, got: {self.source} and {other.source}"
            )

        if (self.target_end_pos - 1 != other.target_begin_pos) or (
            self.source_end_pos - 1 != other.source_begin_pos
        ):
            raise ValueError(
                "Added chains must have one common position"
                "on the end of left chain and beginning of the right chain. "
                f"Got chains:\n{self}\nand\n{other}"
            )

        if self.all_likelihoods[-1] != other.all_likelihoods[0]:
            raise ValueError(
                "Added chains must have one common likelihood"
                "on the end of left chain and beginning of the right chain. "
                f"Got chains:\n{self}\nand\n{other}"
                f"with likelihoods: {other.all_likelihoods[-1]} and {other.all_likelihoods[0]}"
            )

        chain = Chain(
            source=self.source,
            target_begin_pos=self.target_begin_pos,
            source_begin_pos=self.source_begin_pos,
            all_likelihoods=np.concatenate(
                [self.all_likelihoods[:-1], other.all_likelihoods]
            ),
        )

        assert len(chain) == (len(self) + len(other) - 1)
        chain._begin_skips = self._begin_skips
        chain._end_skips = self._end_skips

        return chain

    def to_dict(self) -> dict:
        return {
            "target_begin_pos": self.target_begin_pos,
            "source_begin_pos": self.source_begin_pos,
            "all_likelihoods": self.all_likelihoods.tolist(),
            "source": self.source,
        }

    @staticmethod
    def from_dict(d: dict) -> "Chain":
        return Chain(
            target_begin_pos=d["target_begin_pos"],
            source_begin_pos=d["source_begin_pos"],
            all_likelihoods=np.array(d["all_likelihoods"]),
            source=d["source"],
        )

    def append_end(self, likelihood: float) -> None:
        """Appends significant likelihood to the chain"""
        self.all_likelihoods = np.append(self.all_likelihoods, likelihood)
        self._end_skips = 0

    def skip_end(self, likelihood: float) -> None:
        """Appends insignificant likelihood to the chain"""
        self.all_likelihoods = np.append(self.all_likelihoods, likelihood)
        self._end_skips += 1
        if len(self) == 0:
            self._begin_skips += 1

    def reverse(self) -> Chain:
        rev_chain = Chain(
            source=self.source,
            target_begin_pos=self.target_begin_pos - len(self) + 1,
            source_begin_pos=self.source_begin_pos - len(self) + 1,
            all_likelihoods=np.flip(self.all_likelihoods),
            parent=self,
        )

        rev_chain._begin_skips, rev_chain._end_skips = (
            self._end_skips,
            self._begin_skips,
        )

        assert rev_chain.target_end_pos == self.target_begin_pos + 1
        return rev_chain

    def trim(self):
        """Trims insignificant likelihoods from the chain"""
        end_trim = len(self) - self._end_skips
        self.all_likelihoods = self.all_likelihoods[self._begin_skips : end_trim]
        self.target_begin_pos += self._begin_skips
        self._begin_skips = 0
        self._end_skips = 0

    def trim_copy(self) -> Chain:
        """Produces a chain without insignificant likelihoods on the ends"""
        obj = copy.deepcopy(self)
        obj.trim()
        obj.parent = self
        return obj

    def get_target_token_positions(self) -> Set[int]:
        return set(range(self.target_begin_pos, self.target_end_pos))

    def get_source_token_positions(self) -> Set[int]:
        return set(range(self.source_begin_pos, self.source_end_pos))

    def get_score(self):
        # log2(2 + len) * ((lik_h_0 * ... * lik_h_len) ^ 1 / len)   = score
        l = np.exp(np.log(self.likelihoods).mean())
        score = l * (len(self) ** 2)
        return score

    @staticmethod
    def generate_chains(
        source_likelihoods: torch.Tensor,
        source_name: str,
        target_token_ids: List[int],
        target_start_pos: int,
    ) -> List[Chain]:
        """Generates chains of tokens with the same source

        Parameters
        ----------
        source_likelihoods : torch.Tensor
            inferred from the source text likelihoods for the tokens

        source_name : str
            name of the source, doesn't affect generation, all produced chains will have this name

        target_token_ids : List[int]
            token ids of the target text

        target_start_pos : int
            position from where to start building chains
        """

        result_chains: List[Chain] = []
        source_len = len(source_likelihoods)

        for source_start_pos in range(0, source_len):
            # FORWARD CHAINING:
            chain = Chain(source_name, target_start_pos, source_start_pos)
            skips = 0
            shift_upper_bound = min(
                source_len - source_start_pos, len(target_token_ids) - target_start_pos
            )
            for shift in range(0, shift_upper_bound):
                target_pos = target_start_pos + shift
                source_pos = source_start_pos + shift

                assert target_pos < len(target_token_ids)
                assert source_pos < source_len

                token_curr_id = target_token_ids[target_pos]
                token_curr_likelihood = source_likelihoods[source_pos][
                    token_curr_id
                ].item()

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
    def generate_chains_bidirectional(
        source_likelihoods: npt.NDArray[np.float64],
        source_name: str,
        target_token_ids: List[int],
        target_start_pos: int,
    ) -> List[Chain]:
        """Generates chains of tokens with the same source

        Parameters
        ----------
        source_likelihoods : npt.NDArray[np.float64]
            inferred from the source text likelihoods for the tokens

        source_name : str
            name of the source, doesn't affect generation, all produced chains will have this name

        target_token_ids : List[int]
            token ids of the target text

        target_start_pos : int
            position from where to start building chains
        """
        result_chains: List[Chain] = []
        source_len = len(source_likelihoods)

        for source_start_pos in range(0, source_len):
            # FORWARD CHAINING:
            forward_chain = Chain(source_name, target_start_pos, source_start_pos)
            forward_chains = []
            skips = 0
            shift_upper_bound = min(
                source_len - source_start_pos, len(target_token_ids) - target_start_pos
            )
            for shift in range(0, shift_upper_bound):
                target_pos = target_start_pos + shift
                source_pos = source_start_pos + shift

                assert target_pos < len(target_token_ids)
                assert source_pos < source_len

                token_curr_id = target_token_ids[target_pos]
                token_curr_likelihood = source_likelihoods[source_pos][
                    token_curr_id
                ].item()

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
            backward_chain = Chain(source_name, target_start_pos, source_start_pos)
            backward_chains = []
            skips = 0
            shift_upper_bound = min(source_start_pos, target_start_pos)
            for shift in range(0, shift_upper_bound):
                target_pos = target_start_pos - shift
                source_pos = source_start_pos - shift

                assert target_pos >= 0
                assert source_pos >= 0

                token_curr_id = target_token_ids[target_pos]
                token_curr_likelihood = source_likelihoods[source_pos][
                    token_curr_id
                ].item()

                if token_curr_likelihood < Chain.likelihood_significance_threshold:
                    backward_chain.skip_end(token_curr_likelihood)
                    skips += 1
                    if skips > Chain.max_skips:
                        break
                else:
                    backward_chain.append_end(token_curr_likelihood)
                    if len(backward_chain) > 0:
                        backward_chains.append(backward_chain.reverse())

            assert backward_chain.target_begin_pos == forward_chain.target_begin_pos

            # COMBINE CHAINS:
            for backward_chain, forward_chain in itertools.product(
                backward_chains, forward_chains
            ):
                if (
                    backward_chain._end_skips + forward_chain._begin_skips
                    > Chain.max_skips
                ):
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
