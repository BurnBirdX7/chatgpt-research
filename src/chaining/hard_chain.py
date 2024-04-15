from __future__ import annotations

import copy
import itertools

import numpy as np
import numpy.typing as npt
import torch
from typing import Optional, List, Set, Dict, Any

from src.chaining.chain import Chain


class HardChain(Chain):
    likelihood_significance_threshold = 1e-5
    DEFAULT_MAX_SKIPS = 2

    def __init__(
        self,
        source: str,
        target_begin_pos: int,
        source_begin_pos: int,
        all_likelihoods: List[float] | npt.NDArray[np.float32] | None = None,
        parent: Chain | List[Chain] | None = None,
        cause: str = "",
        end_skips: int = 0,
        begin_skips: int = 0,
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

        all_likelihoods : List[float] or NDArray[float], optional
            All likelihoods of tokens from target text appearing in source text.
            If no value is supplied, length of 0 is assumed

        parent : Chain or List[Chain], optional
            Parent chain, intended for debug purposes.
            Generally is a chain that was reduced or transformed into this chain
        """
        super().__init__(source, target_begin_pos, source_begin_pos, parent, cause)
        self.all_likelihoods: npt.NDArray[np.float32] = np.array(
            [] if (all_likelihoods is None) else all_likelihoods, dtype=np.float32
        )

        self.begin_skips = begin_skips
        self.end_skips = end_skips

    # ----- #
    # Magic #
    # ----- #

    def __len__(self) -> int:
        """Length of the chain.
        Amount of tokens covered by the chain in target and source texts
        """
        return len(self.all_likelihoods)

    def __eq__(self, other: HardChain | None) -> bool:
        if other is None:
            return False

        if not isinstance(other, HardChain):
            return False

        return (
            self.source == other.source
            and self.target_begin_pos == other.target_begin_pos
            and self.source_begin_pos == other.source_begin_pos
            and self.begin_skips == other.begin_skips
            and self.end_skips == other.end_skips
            and np.array_equal(self.all_likelihoods, other.all_likelihoods)
        )

    def __hash__(self):
        return hash(
            (
                self.source,
                self.target_begin_pos,
                self.source_begin_pos,
                self.begin_skips,
                self.end_skips,
                # all_likelihoods as skipped, as it is highly unlikely for 2 chains have different arrays
                #   It's expensive (and unnecessary) to hash the array
            )
        )

    def __add__(self, other: HardChain) -> HardChain:
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
        if not isinstance(other, HardChain):
            return NotImplemented

        if self.source != other.source:
            raise ValueError(f"Added chain must have same source, got: {self.source} and {other.source}")

        if len(self) == 0 or len(other) == 0:
            raise ValueError("Chains can't be empty")

        if (self.target_end_pos - 1 != other.target_begin_pos) or (self.source_end_pos - 1 != other.source_begin_pos):
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

        chain = HardChain(
            source=self.source,
            target_begin_pos=self.target_begin_pos,
            source_begin_pos=self.source_begin_pos,
            all_likelihoods=np.concatenate([self.all_likelihoods[:-1], other.all_likelihoods]),
        )

        assert len(chain) == (len(self) + len(other) - 1)
        chain.begin_skips = self.begin_skips
        chain.end_skips = other.end_skips
        chain.parent = [self, other]

        return chain

    def __str__(self) -> str:
        return (
            f"Chain(\n"
            f"\ttarget: [{self.target_begin_pos};{self.target_end_pos}) ~ {self.get_score()}\n"
            f'\tsource: [{self.source_begin_pos};{self.source_end_pos}) ~ "{self.source}"\n'
            f"\tlen: {len(self)}  [sign len: {len(self.significant_likelihoods)}]\n"
            f"\tparent: {self.parent_str()}\n"
            f"\tcause: {self.cause}\n"
            f"\tbegin_skips: {self.begin_skips}, end_skips: {self.end_skips}"
            f")"
        )

    def __repr__(self) -> str:
        return (
            f"Chain("
            f"target_begin_pos={self.target_begin_pos}, "
            f"source_begin_pos={self.source_begin_pos,}"
            f"all_likelihoods={self.all_likelihoods!r}"
            f"source={self.source!r}, "
            f"parent={self.parent!r}, "
            f"cause={self.cause!r}, "
            f"begin_skips={self.begin_skips}, "
            f"end_skips={self.end_skips}"
            f")"
        )

    # --------- #
    # Positions #
    # --------- #

    @property
    def target_end_pos(self) -> int:
        """Last (exclusive) token position of the chain in the target text"""
        return self.target_begin_pos + len(self)

    @property
    def source_end_pos(self) -> int:
        """Last (exclusive) token position of the chain in matched source text"""
        return self.source_begin_pos + len(self)

    def source_matches(self) -> Dict[int, int | None]:
        return {
            t: s
            for t, s in zip(
                range(self.target_begin_pos, self.target_end_pos), range(self.source_begin_pos, self.source_end_pos)
            )
        }

    def get_target_token_positions(self) -> Set[int]:
        return set(range(self.target_begin_pos, self.target_end_pos))

    def get_source_token_positions(self) -> Set[int]:
        return set(range(self.source_begin_pos, self.source_end_pos))

    # ---------------- #
    # Skip information #
    # ---------------- #

    @property
    def target_begin_skips(self) -> int:
        return self.begin_skips

    @property
    def target_end_skips(self) -> int:
        return self.end_skips

    @property
    def source_begin_skips(self) -> int:
        return self.begin_skips

    @property
    def source_end_skips(self) -> int:
        return self.end_skips

    # ----------- #
    # Likelihoods #
    # ----------- #

    @property
    def significant_likelihoods(self) -> npt.NDArray[np.float32]:
        return self.all_likelihoods[self.all_likelihoods >= HardChain.likelihood_significance_threshold]

    def get_target_likelihood(self, target_pos: int) -> float:
        if target_pos < self.target_begin_pos or target_pos >= self.target_end_pos:
            return -1.0

        return float(self.all_likelihoods[target_pos - self.target_begin_pos])

    def get_score(self) -> float:
        # log2(2 + len) * ((lik_h_0 * ... * lik_h_len) ^ 1 / len)   = score
        g_mean = np.exp(np.log(self.significant_likelihoods).mean())
        score = g_mean * (len(self) ** 2)
        return score

    def significant_len(self) -> int:
        return len(self.significant_likelihoods)

    # ------------- #
    # Serialization #
    # ------------- #

    def to_dict(self) -> dict:
        # Parent and attachment aren't saved
        return {
            "target_begin_pos": self.target_begin_pos,
            "source_begin_pos": self.source_begin_pos,
            "all_likelihoods": self.all_likelihoods.tolist(),
            "source": self.source,
            "begin_skips": self.begin_skips,
            "end_skips": self.end_skips,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HardChain":
        return HardChain(
            target_begin_pos=d["target_begin_pos"],
            source_begin_pos=d["source_begin_pos"],
            all_likelihoods=np.array(d["all_likelihoods"]),
            source=d["source"],
            begin_skips=d["begin_skips"],
            end_skips=d.get("end_skips", d["end_skips:"]),
        )

    # --------- #
    # Expansion #
    # --------- #

    def append_end(self, likelihood: float) -> None:
        """Appends significant likelihood to the chain"""
        assert likelihood >= HardChain.likelihood_significance_threshold
        self.all_likelihoods = np.append(self.all_likelihoods, np.float32(likelihood))
        self.end_skips = 0

    def skip_end(self, likelihood: float) -> None:
        """Appends insignificant likelihood to the chain"""
        assert likelihood < HardChain.likelihood_significance_threshold
        self.end_skips += 1
        if len(self.significant_likelihoods) == 0:  # No significant likelihoods encountered yet
            self.begin_skips += 1
        self.all_likelihoods = np.append(self.all_likelihoods, np.float32(likelihood))

    def reverse(self) -> HardChain:
        rev_chain = HardChain(
            source=self.source,
            target_begin_pos=self.target_begin_pos - len(self) + 1,
            source_begin_pos=self.source_begin_pos - len(self) + 1,
            all_likelihoods=np.flip(self.all_likelihoods, axis=0),
            parent=self,
        )

        rev_chain.begin_skips, rev_chain.end_skips = (
            self.end_skips,
            self.begin_skips,
        )

        assert rev_chain.target_end_pos == self.target_begin_pos + 1
        return rev_chain

    class _ChainExpansionDirection: ...

    FORWARD_DIRECTION = _ChainExpansionDirection()
    BACKWARD_DIRECTION = _ChainExpansionDirection()

    @classmethod
    def expand_chain(
        cls,
        expansion_direction: _ChainExpansionDirection,
        starting_target_pos: int,
        target_token_ids: List[int],
        starting_source_pos: int,
        source_likelihoods: npt.NDArray[np.float32],
        source: str,
        skip_limit: int = DEFAULT_MAX_SKIPS,
    ) -> List[HardChain]:

        skips: int = 0

        if expansion_direction == cls.FORWARD_DIRECTION:
            shift_upper_bound = min(
                len(target_token_ids) - starting_target_pos, len(source_likelihoods) - starting_source_pos
            )
            shift_multiplier = 1
            reverse = False
        else:
            shift_upper_bound = min(starting_source_pos, starting_target_pos)
            shift_multiplier = -1
            reverse = True

        chain = HardChain(source, starting_target_pos, starting_source_pos)
        result_chains = []
        for shift in range(0, shift_upper_bound):
            target_pos = starting_target_pos + (shift_multiplier * shift)
            source_pos = starting_source_pos + (shift_multiplier * shift)

            assert 0 <= target_pos < len(target_token_ids)
            assert 0 <= source_pos < len(source_likelihoods)

            token_curr_id = target_token_ids[target_pos]
            token_curr_likelihood = source_likelihoods[source_pos][token_curr_id].item()

            if token_curr_likelihood < HardChain.likelihood_significance_threshold:
                chain.skip_end(token_curr_likelihood)
                skips += 1
                if skips > skip_limit:
                    break
            else:
                chain.append_end(token_curr_likelihood)
                if len(chain) > 1 and not reverse:
                    result_chains.append(copy.copy(chain))
                elif len(chain) > 1:
                    result_chains.append(chain.reverse())

        return result_chains

    @classmethod
    def generate_chains(
        cls,
        source_likelihoods: npt.NDArray[np.float32],
        source_name: str,
        target_token_ids: List[int],
        target_start_pos: int,
    ) -> List[HardChain]:
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

        result_chains: List[HardChain] = []
        source_len = len(source_likelihoods)

        for source_start_pos in range(0, source_len):
            # FORWARD CHAINING:
            result_chains += HardChain.expand_chain(
                cls.FORWARD_DIRECTION,
                target_start_pos,
                target_token_ids,
                source_start_pos,
                source_likelihoods,
                source_name,
            )

        return result_chains

    @classmethod
    def generate_chains_bidirectional(
        cls,
        source_likelihoods: npt.NDArray[np.float32],
        source_name: str,
        target_token_ids: List[int],
        target_start_pos: int,
    ) -> List[HardChain]:
        """Generates chains of tokens with the same source

        Parameters
        ----------
        source_likelihoods : npt.NDArray[np.float32]
            inferred from the source text likelihoods for the tokens

        source_name : str
            name of the source, doesn't affect generation, all produced chains will have this name

        target_token_ids : List[int]
            token ids of the target text

        target_start_pos : int
            position from where to start building chains
        """
        result_chains: Set[HardChain] = set()
        source_len = len(source_likelihoods)

        for source_start_pos in range(0, source_len):
            # FORWARD CHAINING:
            forward_chains = HardChain.expand_chain(
                cls.FORWARD_DIRECTION,
                target_start_pos,
                target_token_ids,
                source_start_pos,
                source_likelihoods,
                source_name,
            )

            # BACKWARD CHAINING:
            backward_chains = HardChain.expand_chain(
                cls.BACKWARD_DIRECTION,
                target_start_pos,
                target_token_ids,
                source_start_pos,
                source_likelihoods,
                source_name,
            )

            # COMBINE CHAINS:
            for backward_chain, forward_chain in itertools.product(backward_chains, forward_chains):
                if backward_chain.end_skips + forward_chain.begin_skips > HardChain.DEFAULT_MAX_SKIPS:
                    # Reject chain combinations with a large gap in the middle
                    continue

                chain: HardChain = backward_chain + forward_chain
                result_chains.add(chain)

            # Add uni-directional chains to the set
            for chain in forward_chains:
                result_chains.add(chain)
            for chain in backward_chains:
                result_chains.add(chain)

        return list(result_chains)
