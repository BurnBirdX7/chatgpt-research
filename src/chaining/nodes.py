from typing import List, Dict, Any, Set, Callable, Tuple

import numpy as np
import numpy.typing as npt

from ..config import EmbeddingBuilderConfig
from ..pipeline import BaseNode
from .descriptors import ChainListDescriptor, Pos2ChainMappingDescriptor
from .elastic_chain import ElasticChain

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
    def chaining_func(self) -> Callable[[npt.NDArray[np.float32], str, List[int], int], List[ElasticChain]]:
        if self.use_bidirectional_chaining:
            return ElasticChain.generate_chains_bidirectional
        else:
            return ElasticChain.generate_chains

    def process(
        self,
        input_text: str,
        sources: List[List[str]],
        source_likelihoods: Dict[str, npt.NDArray[np.float32]],
    ) -> List[ElasticChain]:
        """
        Parameters
        ----------
        input_text : str
            Text that was supplied to the pipeline
        sources
        source_likelihoods

        Returns
        -------
        ElasticChain

        """
        tokenizer = self.eb_config.tokenizer

        input_token_ids = tokenizer.encode(input_text, add_special_tokens=False)
        result_chains = []
        for token_pos, (token_id, token_sources) in enumerate(zip(input_token_ids, sources)):
            self.logger.debug(f"position: {token_pos + 1} / {len(input_token_ids)}")

            for source in set(token_sources):
                likelihoods = source_likelihoods[source]
                self.logger.debug(f"\tsequence length: {len(likelihoods)}, token id: {token_id}, source: {source}")
                result_chains += self.chaining_func(
                    likelihoods,
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

    def process(self, chains: List[ElasticChain]) -> List[ElasticChain]:
        self.logger.debug(f"Chain count: {len(chains)}")
        filtered_chains: List[ElasticChain] = []
        marked_positions: Set[int] = set()  # positions that are marked with some source

        chains = [chain for chain in chains if chain.significant_len() > 1]

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


class AttachMetaData(BaseNode):

    def __init__(self, name: str):
        super().__init__(name, [list, list, dict], ChainListDescriptor())

    @staticmethod
    def get_texts(token_list: List[str], beg: int, end: int) -> Tuple[str, str, str]:
        text = "".join(token_list[beg:end])
        pre = "".join(token_list[max(beg - 3, 0) : beg])
        post = "".join(token_list[end : end + 3])
        return text, pre, post

    def process(
        self, chains: List[ElasticChain], target_tokens: List[str], source_tokens_dict: Dict[str, List[str]]
    ) -> Any:
        for chain in chains:
            source_tokens = source_tokens_dict[chain.source]
            chain.attachment["source_tokens"] = source_tokens[chain.source_begin_pos : chain.source_end_pos]
            chain.attachment["source_text"] = AttachMetaData.get_texts(
                source_tokens, chain.source_begin_pos, chain.source_end_pos
            )
            chain.attachment["target_text"] = AttachMetaData.get_texts(
                target_tokens, chain.target_begin_pos, chain.target_end_pos
            )

        return chains


class Pos2ChainMapNode(BaseNode):
    """
    Converts a list of NON-INTERSECTING chains into mapping (pos -> chain)
    """

    def __init__(self, name: str):
        super().__init__(name, [list], Pos2ChainMappingDescriptor())

    def process(self, chains: List[ElasticChain]) -> Dict[int, ElasticChain]:
        pos2chain: Dict[int, ElasticChain] = {}
        for i, chain in enumerate(chains):
            for pos in chain.get_target_token_positions():
                pos2chain[pos] = chain

        return pos2chain
