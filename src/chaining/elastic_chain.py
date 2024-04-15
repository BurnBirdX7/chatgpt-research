from __future__ import annotations

import copy
import itertools

import numpy as np
import numpy.typing as npt
import torch
from typing import List, Set, Dict, Any, Tuple

from src.chaining.chain import Chain


class ElasticChain(Chain):
    likelihood_significance_threshold = 0.01
    DEFAULT_MAX_SKIPS = 4

    def __init__(
        self,
        source: str,
        target_begin_pos: int,
        source_begin_pos: int,
        likelihoods: List[float] | npt.NDArray[np.float32] | None = None,
        target_mask: List[bool | int | float] | npt.NDArray[np.bool_] | None = None,
        source_mask: List[bool | int | float] | npt.NDArray[np.bool_] | None = None,
        parent: ElasticChain | List[ElasticChain] | None = None,
        cause: str = "",
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

        likelihoods : List[float] or NDArray[float], optional
            All likelihoods of tokens from target text appearing in source text.
            If no value is supplied, length of 0 is assumed

        target_mask : List[float] or NDArray[float], optional
            Must be provided if likelihoods are provided

        source_mask: List[bool | int | float] | npt.NDArray[np.bool_], optional
            Must be provided if likelihoods are provided

        parent : Chain or List[Chain], optional
            Parent chain, intended for debug purposes.
            Generally is a chain that was reduced or transformed into this chain

        cause : str, default = ""
            If parent is present, cause of separation

        """
        super().__init__(source, target_begin_pos, source_begin_pos, parent, cause)

        if likelihoods is not None and (target_mask is None or source_mask is None):
            raise ValueError("If likelihoods are provided, masks should be provided too")

        self.target_begin_pos: int = target_begin_pos
        self.source_begin_pos: int = source_begin_pos

        self.likelihoods: npt.NDArray[np.float32] = np.array(
            [] if likelihoods is None else likelihoods, dtype=np.float32
        )

        self.target_mask: npt.NDArray[np.bool_] = np.array(
            target_mask if target_mask is not None else [], dtype=np.bool_
        )
        self.source_mask: npt.NDArray[np.bool_] = np.array(
            source_mask if source_mask is not None else [], dtype=np.bool_
        )

        self.source = source
        self.parent: ElasticChain | List[ElasticChain] | None = parent
        self.cause: str | None = cause
        self.attachment: Dict[str, Any] = {}

    @property
    def target_end_pos(self) -> int:
        """Last (exclusive) token position of the chain in the target text"""
        return self.target_begin_pos + len(self.target_mask)

    @property
    def source_end_pos(self) -> int:
        """Last (exclusive) token position of the chain in matched source text"""
        return self.source_begin_pos + len(self.source_mask)

    @staticmethod
    def _end_skips(mask: npt.NDArray[np.bool_]) -> int:
        skips: int = 0
        while skips < len(mask) and not mask[-skips - 1]:
            skips += 1
        return skips

    @property
    def target_end_skips(self):
        return ElasticChain._end_skips(self.target_mask)

    @property
    def target_begin_skips(self):
        return ElasticChain._begin_skips(self.target_mask)

    @property
    def source_end_skips(self):
        return ElasticChain._end_skips(self.source_mask)

    @property
    def source_begin_skips(self):
        return ElasticChain._begin_skips(self.source_mask)

    @staticmethod
    def _begin_skips(mask: npt.NDArray[np.bool_]) -> int:
        skips: int = 0
        while skips < len(mask) and not mask[skips]:
            skips += 1
        return skips

    @property
    def significant_likelihoods(self) -> npt.NDArray[np.float32]:
        """Significant likelihoods"""
        return self.likelihoods[self.target_mask]

    def get_target_likelihood(self, target_pos: int) -> float:
        if target_pos < self.target_begin_pos or target_pos >= self.target_end_pos:
            raise ValueError(
                f"Illegal position, expected in range [{self.target_begin_pos};{self.target_end_pos}), "
                f"got: {target_pos}"
            )

        return self.likelihoods[target_pos - self.target_begin_pos].item()

    def __eq__(self, other: ElasticChain | None) -> bool:
        if other is None:
            return False

        if not isinstance(other, ElasticChain):
            return False

        return (
            self.source == other.source
            and self.target_begin_pos == other.target_begin_pos
            and self.source_begin_pos == other.source_begin_pos
            and np.array_equal(self.target_mask, other.target_mask)
            and np.array_equal(self.source_mask, other.source_mask)
            and np.array_equal(self.likelihoods, other.likelihoods)
        )

    def __hash__(self):
        return hash(
            (
                self.source,
                self.target_begin_pos,
                self.source_begin_pos,
                len(self.target_mask),
                len(self.source_mask),
                # likelihoods are skipped, as it is highly unlikely for 2 chains have different arrays
                #   It's expensive (and unnecessary) to hash the array
            )
        )

    def __len__(self) -> int:
        """Length of the chain.
        Amount of tokens covered by the chain in target and source texts
        """
        return len(self.target_mask)

    def significant_len(self):
        return self.target_mask.sum()

    def __str__(self) -> str:
        cause =  f"lost[{self.cause}]" if self.cause != "" else ""

        return (
            f"Chain(\n"
            f"\ttarget: [{self.target_begin_pos};{self.target_end_pos}) ~ {self.get_score()}\n"
            f'\tsource: [{self.source_begin_pos};{self.source_end_pos}) ~ "{self.source}"\n'
            f"\tlen: {len(self)}  [sign len: {len(self.significant_likelihoods)}]\n"
            f"\tparent: {self.parent_str()}\n"
            f"\tcause: {cause}"
            f")"
        )

    def __repr__(self) -> str:
        return (
            f"Chain("
            f"target_begin_pos={self.target_begin_pos}, "
            f"source_begin_pos={self.source_begin_pos,}"
            f"likelihoods={self.likelihoods!r}, "
            f"source_mask={self.source_mask!r}, "
            f"target_mask={self.target_mask!r}, "
            f"source={self.source!r}, "
            f"parent={self.parent!r}, "
            f"cause={self.cause!r}"
            f")"
        )

    def __add__(self, other: ElasticChain) -> ElasticChain:
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
        if not isinstance(other, ElasticChain):
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

        if self.likelihoods[-1] != other.likelihoods[0]:
            raise ValueError(
                "Added chains must have one common likelihood"
                "on the end of left chain and beginning of the right chain. "
                f"Got chains:\n{self}\nand\n{other}"
                f"with likelihoods: {other.likelihoods[-1]} and {other.likelihoods[0]}"
            )

        chain = ElasticChain(
            source=self.source,
            target_begin_pos=self.target_begin_pos,
            source_begin_pos=self.source_begin_pos,
            likelihoods=np.concatenate([self.likelihoods[:-1], other.likelihoods]),
            source_mask=np.concatenate([self.source_mask[:-1], other.source_mask]),
            target_mask=np.concatenate([self.target_mask[:-1], other.target_mask]),
            parent=[self, other],
            cause="add",
        )

        assert len(chain) == (len(self) + len(other) - 1)

        return chain

    def to_dict(self) -> dict:
        # Parent and attachment aren't saved
        return {
            "target_begin_pos": self.target_begin_pos,
            "source_begin_pos": self.source_begin_pos,
            "likelihoods": self.likelihoods.tolist(),
            "source": self.source,
            "source_len": len(self.source_mask),
            "source_skipped": np.flatnonzero(~self.source_mask).tolist(),
            "target_len": len(self.target_mask),
            "target_skipped": np.flatnonzero(~self.target_mask).tolist(),
            "cause": self.cause,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ElasticChain":
        source_len = d["source_len"]
        source_mask = np.ones(source_len, dtype=np.bool_)
        source_mask[d["source_skipped"]] = False

        target_len = d["target_len"]
        target_mask = np.ones(target_len, dtype=np.bool_)
        target_mask[d["target_skipped"]] = False

        return ElasticChain(
            source=d["source"],
            target_begin_pos=d["target_begin_pos"],
            source_begin_pos=d["source_begin_pos"],
            likelihoods=np.array(d["likelihoods"]),
            source_mask=source_mask,
            target_mask=target_mask,
            cause=d["cause"],
        )

    def copy(self, cause: str | None = "copy") -> ElasticChain:
        c = copy.copy(self)
        if cause is not None:
            c.parent = self
            c.cause = cause
        return c

    def expand_forward(self, likelihood: float):
        self.likelihoods = np.append(self.likelihoods, np.float32(likelihood))
        self.target_mask = np.append(self.target_mask, True)
        self.source_mask = np.append(self.source_mask, True)

    def expand_backward(self, likelihood: float):
        self.likelihoods = np.append(np.float32(likelihood), self.likelihoods)
        self.target_mask = np.append(True, self.target_mask)
        self.source_mask = np.append(True, self.source_mask)
        self.target_begin_pos -= 1
        self.source_begin_pos -= 1

    def skip_forward(self, likelihood: float) -> Tuple[ElasticChain, ElasticChain]:
        """
        Marks following token-positions as skipped

        Parameters
        ----------
        likelihood

        Returns
        -------
        Tuple[ElasticChain, ElasticChain]
            Tuple of chains, where token-pos is skipped only in target text and where its skipped only in source text

        Examples
        --------

        Say we tokens in target and source texts: ``[t0,t1,t2,t3,t4,t5]`` and ``[s0,s1,s2,s3,s4,s5]``

        And we matched these token sequences up to 't3' and 's3' tokens, so we have TokenChain:
        >>> the_chain = ElasticChain (
        ...    source="some_source",
        ...    target_begin_pos=17,
        ...    source_begin_pos=13,
        ...    likelihoods=[0.9, 0.8, 0.9],
        ...    target_mask=[1, 1, 1],
        ...    source_mask=[1, 1, 1],
        ... )
        That matches sequences [t0, t1, t2] and [s0, s1, s2]

        Let's skip next position:
        >>> fork1, fork2 = the_chain.skip_forward(0.1) # Our tokens are identical, so likelihood is likely to be high :)

        ``the_chain`` is now matches sequences ``[t0,t1,t2,t3]`` and ``[s0,s1,s2,s3]``,
            but ``t3`` and ``s3`` tokens are skipped
        It can describe a situation where tokens ``t0``-``t2`` are likely to appear in place of ``s0``-``s2``
                but ``t3`` isn't likely to appear instead of ``s3``.

        Simplified: where ``t0``-``t2`` are equal to ``s0``-``s2``, but ``t3`` and ``s3`` are different

        The method produces 2 forks, that skip positions only in one place (target tokens or source tokens)

        It would match [a, b, c, x, e] and [a, b, c, y, e] sequences pretty good

        >>> fork1.expand_forward(0.9)
        ... fork2.expand_forward(0.8)

        After that ``fork1`` will match target sequence ``[t0,t1,t2,t3,t4]`` to ``[s0,s1,s2,s3]``,
            where ``t3`` is skipped and ``t4`` is matched against ``s3``,

            ``t3`` has likelihood of `0.0` and matches against `None`

            It would match sequences [a, b, c, d, e, f] and [a, b, c, e, f]

        And ``fork2`` will match target sequence ``[t0,t1,t2,t3]`` to ``[s0,s1,s2,s3,s4]``,
            where ``s3`` is skipped and ``t3`` is matched against ``s4``

            It would match sequences [a, b, c, e, f] and [a, b, c, d, e, f]
        """
        target_skip = self.copy("fork/target")
        target_skip.target_mask = np.append(target_skip.target_mask, False)
        target_skip.likelihoods = np.append(target_skip.likelihoods, 0.0)

        source_skip = self.copy("fork/source")
        source_skip.source_mask = np.append(source_skip.source_mask, False)

        self.source_mask = source_skip.source_mask
        self.target_mask = target_skip.target_mask
        self.likelihoods = np.append(self.likelihoods, np.float32(likelihood))

        return target_skip, source_skip

    def skip_backward(self, likelihood: float) -> Tuple[ElasticChain, ElasticChain]:
        """
        Reversed versio  of ``skip_forward``

        See Also
        --------
        skip_forward : skips token in front of a sequence, has detailed docstring
        """
        target_skip = self.copy("fork/target")
        target_skip.target_mask = np.append(False, target_skip.target_mask)
        target_skip.likelihoods = np.append(0.0, target_skip.likelihoods)

        source_skip = self.copy("fork/source")
        source_skip.source_mask = np.append(False, source_skip.source_mask)

        self.source_mask = source_skip.source_mask
        self.target_mask = target_skip.target_mask
        self.likelihoods = np.append(np.float32(likelihood), self.likelihoods)

        return target_skip, source_skip

    def source_positions(self) -> List[int | None]:
        """
        Returns relative source positions that
        """
        assert self.target_mask.sum() == self.source_mask.sum()

        s_pos = 0

        lst: List[int | None] = [None] * len(self)
        for i in range(len(lst)):
            if not self.target_mask[i]:
                continue

            # Ship all source skips
            while not self.source_mask[s_pos]:
                s_pos += 1

            lst[i] = s_pos
            s_pos += 1

        return lst

    def _trim(self, mask: npt.NDArray, *targets: str) -> int:
        begin_trim = ElasticChain._begin_skips(mask)
        end_trim = len(mask) - ElasticChain._end_skips(mask)

        for target in targets:
            old_value = getattr(self, target)
            setattr(self, target, old_value[begin_trim:end_trim])

        return begin_trim

    def trim(self):
        self.target_begin_pos += self._trim(self.target_mask, "target_mask", "likelihoods")
        self.source_begin_pos += self._trim(self.source_mask, "source_mask")

    def trim_copy(self) -> ElasticChain:
        """Produces a chain without insignificant likelihoods on the ends"""
        obj = self.copy("trim")
        obj.trim()
        return obj

    def get_target_token_positions(self) -> Set[int]:
        return set(range(self.target_begin_pos, self.target_end_pos))

    def get_source_token_positions(self) -> Set[int]:
        return set(range(self.source_begin_pos, self.source_end_pos))

    def get_score(self) -> float:
        # log2(2 + len) * ((lik_h_0 * ... * lik_h_len) ^ 1 / len)   = score
        g_mean = np.exp(np.log(self.significant_likelihoods).mean())
        score = g_mean * (len(self) ** 2)
        return score

    def reached_skip_limit(self, skip_limit: int) -> bool:
        def reached_limit(mask):
            return len(mask) >= skip_limit and not (mask[-skip_limit:].any() and mask[:skip_limit].any())

        return reached_limit(self.target_mask) or reached_limit(self.source_mask)

    class _ChainExpansionDirection: ...

    FORWARD_DIRECTION = _ChainExpansionDirection()
    BACKWARD_DIRECTION = _ChainExpansionDirection()

    def _expand_chain(
        self: ElasticChain,
        expansion_direction: _ChainExpansionDirection,
        target_token_ids: List[int],
        source_likelihoods: npt.NDArray[np.float32],
        skip_limit: int = DEFAULT_MAX_SKIPS,
        depth: int = 0,
    ) -> List[ElasticChain]:

        if self.reached_skip_limit(skip_limit):
            return []

        # SETUP:
        if expansion_direction == self.FORWARD_DIRECTION:
            expand_f = ElasticChain.expand_forward
            skip_f = ElasticChain.skip_forward
            shift_upper_bound = min(
                len(source_likelihoods) - self.source_end_pos, len(target_token_ids) - self.target_end_pos
            )
            target_start_pos = self.target_end_pos
            source_start_pos = self.source_end_pos
            shift_multiplier = 1

        else:
            expand_f = ElasticChain.expand_backward
            skip_f = ElasticChain.skip_backward
            shift_upper_bound = min(self.source_begin_pos, self.target_begin_pos)
            target_start_pos = self.target_begin_pos - 1
            source_start_pos = self.source_begin_pos - 1
            shift_multiplier = -1

        # GENERATION:
        result_chains = []
        for shift in range(0, shift_upper_bound):
            target_pos = target_start_pos + (shift_multiplier * shift)
            source_pos = source_start_pos + (shift_multiplier * shift)

            assert 0 <= target_pos < len(target_token_ids)
            assert 0 <= source_pos < len(source_likelihoods)

            token_curr_id = target_token_ids[target_pos]
            token_curr_likelihood = source_likelihoods[source_pos][token_curr_id].item()

            if token_curr_likelihood < ElasticChain.likelihood_significance_threshold:
                if depth == 0 and shift == 0:
                    return []

                fork1, fork2 = skip_f(self, token_curr_likelihood)
                result_chains += fork1._expand_chain(
                    expansion_direction, target_token_ids, source_likelihoods, skip_limit, depth + 1
                )
                result_chains += fork2._expand_chain(
                    expansion_direction,
                    target_token_ids,
                    source_likelihoods,
                    skip_limit,
                    depth + 1,
                )
                break

            else:
                expand_f(self, token_curr_likelihood)
                result_chains.append(self.copy(None))

        return result_chains

    @classmethod
    def generate_chains(
        cls,
        source_likelihoods: npt.NDArray[np.float32],
        source_name: str,
        target_token_ids: List[int],
        target_start_pos: int,
    ) -> List[ElasticChain]:
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

        result_chains: List[ElasticChain] = []
        source_len = len(source_likelihoods)

        for source_start_pos in range(0, source_len):
            # FORWARD CHAINING:
            chain = ElasticChain(source_name, target_start_pos, source_start_pos)
            result_chains += chain._expand_chain(
                cls.FORWARD_DIRECTION,
                target_token_ids,
                source_likelihoods,
            )

        return result_chains

    @classmethod
    def generate_chains_bidirectional(
        cls,
        source_likelihoods: npt.NDArray[np.float32],
        source_name: str,
        target_token_ids: List[int],
        target_start_pos: int,
    ) -> List[ElasticChain]:
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
        result_chains: Set[ElasticChain] = set()
        source_len = len(source_likelihoods)

        for source_start_pos in range(0, source_len):
            # FORWARD CHAINING:
            forward_chains = ElasticChain(source_name, target_start_pos, source_start_pos)._expand_chain(
                cls.FORWARD_DIRECTION, target_token_ids, source_likelihoods
            )

            # BACKWARD CHAINING:
            backward_chains = ElasticChain(source_name, target_start_pos, source_start_pos)._expand_chain(
                cls.BACKWARD_DIRECTION, target_token_ids, source_likelihoods
            )

            # COMBINE CHAINS:
            for backward_chain, forward_chain in itertools.product(backward_chains, forward_chains):
                if (
                    backward_chain.source_end_skips + forward_chain.source_begin_skips > ElasticChain.DEFAULT_MAX_SKIPS
                    or backward_chain.target_end_skips + forward_chain.target_begin_skips
                    > ElasticChain.DEFAULT_MAX_SKIPS
                ):
                    # Reject chain combinations with a large gap in the middle
                    continue
                try:  # Try adding chain, if fails - skip
                    chain: ElasticChain = backward_chain + forward_chain
                    chain.trim()
                    result_chains.add(chain)
                except ValueError:
                    ...

            # Add uni-directional chains to the set
            for chain in forward_chains:
                result_chains.add(chain)
            for chain in backward_chains:
                result_chains.add(chain)

        return list(result_chains)
