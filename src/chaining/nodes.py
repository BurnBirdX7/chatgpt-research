import logging
import typing as t

import numpy as np
import numpy.typing as npt

from src.config import EmbeddingBuilderConfig
from src.embeddings_builder import NDArrayDescriptor
from src.pipeline import BaseNode, ListDescriptor
from .descriptors import ChainListDescriptor, Pos2ChainMappingDescriptor
from .chain import Chain
from .hard_chain import HardChain
from .wide_chain import WideChain

__all__ = [
    "ChainingNode",
    "FilterOverlappingChainsNode",
    "Pos2ChainMapNode",
    "AttachMetaData",
    "WideChaining",
    "CollectTokenScoreNode",
]


class ChainingNode(BaseNode):
    def __init__(
        self,
        name: str,
        embedding_builder_config: EmbeddingBuilderConfig,
        use_bidirectional_chaining: bool = False,
        chain_class: t.Type[Chain] = HardChain,
    ):
        super().__init__(name, [str, list, dict], ChainListDescriptor())
        self.eb_config = embedding_builder_config
        self.use_bidirectional_chaining = use_bidirectional_chaining
        self.chain_class = chain_class

    @property
    def chaining_func(self) -> t.Callable[[npt.NDArray[np.float32], str, t.List[int], int], t.Sequence[Chain]]:
        if self.use_bidirectional_chaining:
            return self.chain_class.generate_chains_bidirectionally
        else:
            return self.chain_class.generate_chains

    def process(
        self,
        input_text: str,
        sources: t.List[t.List[str]],
        source_likelihoods: t.Dict[str, npt.NDArray[np.float32]],
    ) -> t.List[Chain]:
        """
        Parameters
        ----------
        input_text : str
            Text that was supplied to the pipeline
        sources
        source_likelihoods

        Returns
        -------
        List[Chain]

        """
        tokenizer = self.eb_config.tokenizer

        input_token_ids = tokenizer.encode(input_text, add_special_tokens=False)
        result_chains: t.List[Chain] = []
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


class FilterOverlappingChainsNode(BaseNode):
    """
    Removes intersections between chains giving priority to chains with higher score
    """

    def __init__(self, name: str):
        super().__init__(name, [list], ChainListDescriptor())

    def process(self, chains: t.List[Chain]) -> t.List[Chain]:
        self.logger.debug(f"Chain count: {len(chains)}")
        filtered_chains: t.List[Chain] = []
        marked_positions: t.Set[int] = set()  # positions that are marked with some source

        chains = [chain for chain in chains if chain.significant_len() > 1]

        for chain in sorted(chains, key=lambda x: x.get_score(), reverse=True):
            if chain.target_len() < 2:
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
    def get_texts(token_list: t.List[str], beg: int, end: int) -> t.Tuple[str, str, str]:
        text = "".join(token_list[beg:end])
        pre = "".join(token_list[max(beg - 3, 0) : beg])
        post = "".join(token_list[end : end + 3])
        return text, pre, post

    def process(
        self, chains: t.List[Chain], target_tokens: t.List[str], source_tokens_dict: t.Dict[str, t.List[str]]
    ) -> t.Any:
        for chain in chains:
            source_tokens = source_tokens_dict[chain.source]
            chain.attachment["source_tokens"] = source_tokens
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

    def process(self, chains: t.List[Chain]) -> t.Dict[int, Chain]:
        pos2chain: t.Dict[int, Chain] = {}
        for i, chain in enumerate(chains):
            for pos in chain.get_target_token_positions():
                pos2chain[pos] = chain

        return pos2chain


class WideChaining(BaseNode):

    def __init__(self, name, eb_config: EmbeddingBuilderConfig):
        super().__init__(name, [str, list, dict], ChainListDescriptor())
        self.eb_config = eb_config

    def process(
        self, input_text: str, sources: t.List[t.List[str]], source_likelihoods: t.Dict[str, npt.NDArray[np.float32]]
    ) -> t.List[Chain]:
        tokenizer = self.eb_config.tokenizer
        input_token_ids = tokenizer.encode(input_text, add_special_tokens=False)

        source_chains: t.Dict[str, t.Sequence[Chain]] = {}

        for token_pos, (token_id, token_sources) in enumerate(zip(input_token_ids, sources)):
            self.logger.debug(f"position: {token_pos + 1} / {len(input_token_ids)}")
            for source in token_sources:
                if source in source_chains:
                    continue

                source_chains[source] = WideChain.generate_chains_bidirectionally(
                    source_likelihoods[source], source, input_token_ids, 0  # ignored
                )

        return [chain for chains in source_chains.values() for chain in chains]


class CollectTokenScoreNode(BaseNode):

    @staticmethod
    def default_score(slice_: t.List[float]) -> np.float32:
        if len(slice_) == 0:
            logging.getLogger(__name__).debug("Zero len slice encountered")
            return np.float32(0.0)

        return np.exp(np.log(slice_, dtype=np.float32).sum() / len(slice_))

    def __init__(self, name: str):
        super().__init__(name, [list, list], NDArrayDescriptor())
        self.score_func: t.Callable[[t.List[float]], np.float32] = CollectTokenScoreNode.default_score

    def process(self, tokens: t.List[str], chains: t.List[WideChain]) -> npt.NDArray[np.float32]:
        likelihoods = [[] for _ in tokens]
        for chain in chains:
            for pos in range(chain.target_begin_pos, chain.target_end_pos):
                likelihoods[pos].append(chain.get_target_likelihood(pos))

        arr = np.empty(len(tokens))
        for i in range(len(tokens)):
            arr[i] = self.score_func(likelihoods[i])

        return arr
