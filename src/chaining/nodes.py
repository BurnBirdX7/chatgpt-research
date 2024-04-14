from typing import List, Dict, Any, Set, Callable

import numpy as np
import numpy.typing as npt

from ..config import EmbeddingBuilderConfig
from ..pipeline import BaseNode
from .descriptors import ChainListDescriptor, Pos2ChainMappingDescriptor
from .token_chain import TokenChain

__all__ = ["ChainingNode", "FilterChainsNode", "Pos2ChainMapNode"]


class ChainingNode(BaseNode):
    def __init__(
        self,
        name: str,
        embedding_builder_config: EmbeddingBuilderConfig,
        use_bidirectional_chaining: bool = False,
    ):
        super().__init__(name, [str, list, dict], ChainListDescriptor())
        self.eb_config = embedding_builder_config
        self.use_bidirectional_chaining = use_bidirectional_chaining

    @property
    def chaining_func(self) -> Callable[[npt.NDArray[np.float64], str, List[int], int], List[TokenChain]]:
        if self.use_bidirectional_chaining:
            return TokenChain.generate_chains_bidirectional
        else:
            return TokenChain.generate_chains

    def process(
        self,
        input_text: str,
        sources: List[List[str]],
        source_batched_likelihoods: Dict[str, npt.NDArray[np.float32]],
    ) -> List[TokenChain]:
        """
        Parameters
        ----------
        input_text : str
            Text that was supplied to the pipeline
        sources
        source_batched_likelihoods

        Returns
        -------
        TokenChain

        """
        tokenizer = self.eb_config.tokenizer

        input_token_ids = tokenizer.encode(input_text, add_special_tokens=False)
        result_chains = []
        for token_pos, (token_id, token_sources) in enumerate(zip(input_token_ids, sources)):
            self.logger.debug(f"position: {token_pos + 1} / {len(input_token_ids)}")

            for source in token_sources:
                batched_likelihoods = source_batched_likelihoods[source]
                self.logger.debug(
                    f"\tbatch size: {len(batched_likelihoods)}, " f"token id: {token_id}, " f"source: {source}"
                )

                result_chains += self.chaining_func(
                    batched_likelihoods,
                    source,
                    input_token_ids,
                    token_pos,
                )

        return result_chains


class FilterChainsNode(BaseNode):
    """
    Removes intersections between chains giving priority to chains with higher score
    """

    def __init__(self, name: str):
        super().__init__(name, [list], ChainListDescriptor())

    def process(self, chains: List[TokenChain]) -> List[TokenChain]:
        self.logger.debug(f"Chain count: {len(chains)}")
        filtered_chains: List[TokenChain] = []
        marked_positions: Set[int] = set()  # positions that are marked with some source

        chains = [chain.trim_copy() for chain in chains]
        chains = [chain for chain in chains if len(chain) > 1]

        for chain in sorted(chains, key=lambda x: x.get_score(), reverse=True):
            if len(chain) < 2:
                continue

            positions = chain.get_target_token_positions()
            marked_positions_inside_chain = marked_positions.intersection(positions)
            if len(marked_positions_inside_chain) == 0:
                marked_positions |= positions
                filtered_chains.append(chain)

        self.logger.debug(f"Filtered chains count: {len(filtered_chains)}")
        return filtered_chains


class Pos2ChainMapNode(BaseNode):
    """
    Converts a list of NON-INTERSECTING chains into mapping (pos -> chain)
    """

    def __init__(self, name: str):
        super().__init__(name, [list], Pos2ChainMappingDescriptor())

    def process(self, chains: List[TokenChain]) -> Dict[int, TokenChain]:
        pos2chain: Dict[int, TokenChain] = {}
        for i, chain in enumerate(chains):
            for pos in chain.get_target_token_positions():
                pos2chain[pos] = chain

        return pos2chain
