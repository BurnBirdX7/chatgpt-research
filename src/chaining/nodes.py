from typing import List, Dict, Any, Set

import torch

from ..config import EmbeddingBuilderConfig
from ..pipeline import BaseNode
from .descriptors import ChainListDescriptor, Pos2ChainMappingDescriptor
from .token_chain import Chain

__all__ = ['ChainingNode', 'FilterChainsNode', 'Pos2ChainMapNode']


class ChainingNode(BaseNode):
    def __init__(self, name: str, embedding_builder_config: EmbeddingBuilderConfig):
        super().__init__(name, [str, list, dict], ChainListDescriptor())
        self.eb_config = embedding_builder_config

    def process(self, input_text: str, sources: List[List[str]],
                source_batched_likelihoods: Dict[str, torch.Tensor]) -> Any:
        tokenizer = self.eb_config.tokenizer

        input_token_ids = tokenizer.encode(input_text)
        result_chains = []
        for token_pos, (token_id, token_sources) in enumerate(zip(input_token_ids, sources)):
            self.logger.debug(f"position: {token_pos + 1} / {len(input_token_ids)}")

            for source in token_sources:
                batched_likelihoods = source_batched_likelihoods[source]
                self.logger.debug(f"\tbatch size: {len(batched_likelihoods)}, "
                                  f"token id: {token_id}, "
                                  f"source: {source}")

                for i, passage_likelihoods in enumerate(batched_likelihoods):
                    result_chains += Chain.generate_chains(passage_likelihoods, source,
                                                           input_token_ids, token_pos)

        return result_chains


class FilterChainsNode(BaseNode):
    """
    Removes intersections between chains giving priority to chains with higher score
    """

    def __init__(self, name: str):
        super().__init__(name, [list], ChainListDescriptor())

    def process(self, chains: List[Chain]) -> List[Chain]:
        print("Chain count: ", len(chains))
        filtered_chains: List[Chain] = []
        marked_positions: Set[int] = set()  # positions that are marked with some source
        for chain in sorted(chains, key=lambda x: x.get_score(), reverse=True):
            positions = chain.get_token_positions()
            marked_positions_inside_chain = marked_positions.intersection(positions)
            if len(marked_positions_inside_chain) == 0:
                marked_positions |= positions
                filtered_chains.append(chain)

        print("Filtered chains count: ", len(filtered_chains))
        return filtered_chains


class Pos2ChainMapNode(BaseNode):
    """
    Converts a list of NON-INTERSECTING chains into mapping (pos -> chain)
    """

    def __init__(self, name: str):
        super().__init__(name, [list], Pos2ChainMappingDescriptor())

    def process(self, chains: List[Chain]) -> Dict[int, Chain]:
        pos2chain: Dict[int, Chain] = {}
        for i, chain in enumerate(chains):
            for pos in chain.get_token_positions():
                pos2chain[pos] = chain

        return pos2chain
