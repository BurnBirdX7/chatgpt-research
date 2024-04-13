import copy
import itertools
from typing import Any, List, Dict, Tuple

from transformers import RobertaForMaskedLM

from src import QueryColbertServerNode, Chain
from src.chaining import ChainingNode, FilterChainsNode, Pos2ChainMapNode
from src.chaining.descriptors import ChainListDescriptor
from src.embeddings_builder import (
    EmbeddingsFromTextNode,
    TokenizeTextNode,
    LikelihoodsForMultipleSources,
)
from src.index import IndexFromSourcesNode, SearchIndexNode
from src.pipeline import (
    Pipeline,
    mapping_node,
    DictDescriptor,
    ListDescriptor,
    ComplexDictDescriptor,
    BaseNode,
)
from src.config import ColbertServerConfig, EmbeddingBuilderConfig
from src.pipeline.wrapper_nodes import DictWrapperNode
from src.text_processing import TextProcessingNode, remove_punctuation


@mapping_node(out_descriptor=DictDescriptor())
def FilterDict(dict_: dict, keys: list) -> dict:
    # Keys is a list of lists
    unique_keys = set(itertools.chain.from_iterable(keys))
    return {k: dict_[k] for k in unique_keys}


class AttachMetaData(BaseNode):

    def __init__(self, name: str):
        super().__init__(name, [list, list, dict], ChainListDescriptor())

    @staticmethod
    def get_texts(token_list: List[str], beg: int, end: int) -> Tuple[str, str, str]:
        text = "".join(token_list[beg: end])
        pre = "".join(token_list[max(beg - 3, 0): beg])
        post = "".join(token_list[end: end + 3])
        return text, pre, post

    def process(self, chains: List[Chain], target_tokens: List[str], source_tokens_dict: Dict[str, List[str]]) -> Any:
        """
        Parameters
        ----------
        chains : List[Chain]
        target_tokens : List[str]
        source_tokens_dict : Dict[str, List[str]]

        Returns
        -------
        List[Chain]
        """

        for chain in chains:
            source_tokens = source_tokens_dict[chain.source]
            chain.attachment["source_tokens"] = source_tokens[chain.source_begin_pos:chain.source_end_pos]
            chain.attachment["source_text"] = AttachMetaData.get_texts(
                source_tokens, chain.source_begin_pos, chain.source_end_pos
            )
            chain.attachment["target_text"] = AttachMetaData.get_texts(
                target_tokens, chain.target_begin_pos, chain.target_end_pos
            )

        return chains


def get_coloring_pipeline(name: str = "text-coloring") -> Pipeline:
    # Configs:
    colbert_cfg: ColbertServerConfig = ColbertServerConfig.load_from_env()  # type: ignore
    text_eb_config = EmbeddingBuilderConfig(normalize=True, centroid_file="artifacts/centroid-colbert.npy")  # Default
    chaining_eb_config = EmbeddingBuilderConfig(
        model=RobertaForMaskedLM.from_pretrained("roberta-large").to(text_eb_config.model.device),
        centroid_file="artifacts/centroid-colbert.npy",
    )

    # == Pipeline ==

    # First node strips input of punctuation
    pipeline = Pipeline(TextProcessingNode.new("input-stripped", remove_punctuation), name=name)

    # Node queries all sources that might contain similar text from ColBERT
    pipeline.attach_back(QueryColbertServerNode("all-sources-dict-raw", colbert_cfg))

    # Clear all retrieved texts of punctuation
    pipeline.attach_back(TextProcessingNode.new_for_dicts("all-sources-dict", remove_punctuation))

    # Generate embeddings from sources and build new FAISS index
    pipeline.attach_back(IndexFromSourcesNode("all-sources-index", text_eb_config))

    # Produce embeddings from input
    pipeline.attach(EmbeddingsFromTextNode("input-embeddings", text_eb_config), "input-stripped")

    # Get possible source names for every token-pos of the input text
    pipeline.attach(
        SearchIndexNode("narrowed-sources-list", k=10),
        "all-sources-index",
        "input-embeddings",
    )

    # Narrow dictionary
    pipeline.attach(FilterDict("narrowed-sources-dict"), "all-sources-dict", "narrowed-sources-list")

    # Get likelihoods for all token-positions for given sources
    pipeline.attach_back(LikelihoodsForMultipleSources("source-likelihoods", chaining_eb_config))

    # Build high-likelihood chains
    pipeline.attach(
        ChainingNode("all-chains", chaining_eb_config, True),
        "input-stripped",
        "narrowed-sources-list",
        "source-likelihoods",
    )

    # Filter overlapping chains out
    pipeline.attach_back(FilterChainsNode("filtered-chains"))

    # Map every token-position onto corresponding chain
    pipeline.attach_back(Pos2ChainMapNode("token2chain"))

    return pipeline


def get_extended_coloring_pipeline(name: str = "text-coloring") -> Pipeline:
    # Configs:
    text_eb_config = EmbeddingBuilderConfig(normalize=True, centroid_file="artifacts/centroid-colbert.npy")

    # == Pipeline ==

    pipeline = get_coloring_pipeline(name)

    # (MISC) Input text is split into tokens
    pipeline.attach(
        TokenizeTextNode("input-tokenized", text_eb_config),
        "input-stripped",
        auxiliary=True,
    )

    # (MISC) Split source texts into tokens
    pipeline.attach(
        DictWrapperNode(TokenizeTextNode("narrowed-sources-dict-tokenized", text_eb_config)),
        "narrowed-sources-dict",
        auxiliary=True,
    )

    # Add matched source texts to filetered chains
    pipeline.attach(
        AttachMetaData("chains-with-text"),
        "filtered-chains",
        "input-tokenized",
        "narrowed-sources-dict-tokenized",
        auxiliary=True,
    )

    return pipeline


if __name__ == "__main__":
    print("This scripts isn't runnable, it provides `get_coloring_pipeline` function")
