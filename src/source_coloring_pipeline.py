from __future__ import annotations

import itertools

from transformers import RobertaForMaskedLM

from . import QueryColbertServerNode, ChainingNode, FilterOverlappingChainsNode
from .chaining.nodes import AttachMetaData
from .config import EmbeddingBuilderConfig, IndexConfig, ColbertServerConfig
from .embeddings_builder import EmbeddingsFromTextNode, LikelihoodsForMultipleSources, TokenizeTextNode
from .index import IndexFromSourcesNode, SearchIndexNode
from .pipeline import Pipeline, mapping_node, DictDescriptor
from .pipeline.wrapper_nodes import DictWrapperNode
from .text_processing import TextProcessingNode, remove_punctuation, remove_wiki_formatting


@mapping_node(out_descriptor=DictDescriptor())
def FilterDict(dict_: dict, keys: list) -> dict:
    # Keys is a list of lists
    unique_keys = set(itertools.chain.from_iterable(keys))
    return {k: dict_[k] for k in unique_keys}


class SourceColoringPipeline(Pipeline):

    def __init__(self, name, use_bidirectional_chaining: bool = True):
        super().__init__(
            TextProcessingNode.new("input-stripped", remove_punctuation | remove_wiki_formatting), name=name
        )

        # CONFIGS:
        self.colbert_cfg: ColbertServerConfig = ColbertServerConfig.load_from_env()  # type: ignore
        self.text_embeddingBuilder_cfg = EmbeddingBuilderConfig(
            normalize=True, centroid_file="artifacts/centroid-colbert.npy"
        )
        self.chaining_embeddingBuilder_cfg = EmbeddingBuilderConfig(
            model=RobertaForMaskedLM.from_pretrained("roberta-large").to(self.text_embeddingBuilder_cfg.model.device),
            centroid_file="artifacts/centroid-colbert.npy",
        )
        self.source_index_cfg = IndexConfig(faiss_use_gpu=False)

        # Node queries all sources that might contain similar text from ColBERT

        self.attach_back(QueryColbertServerNode("all-sources-dict-raw", self.colbert_cfg))

        # Clear all retrieved texts of punctuation
        self.attach_back(
            TextProcessingNode.new_for_dicts("all-sources-dict", remove_punctuation | remove_wiki_formatting)
        )

        # Generate embeddings from sources and build new FAISS index
        self.attach_back(
            IndexFromSourcesNode("all-sources-index", self.text_embeddingBuilder_cfg, self.source_index_cfg)
        )

        # Produce embeddings from input
        self.attach(EmbeddingsFromTextNode("input-embeddings", self.text_embeddingBuilder_cfg), "input-stripped")

        # Get possible source names for every token-pos of the input text
        self.attach(
            SearchIndexNode("narrowed-sources-list", k=10),
            "all-sources-index",
            "input-embeddings",
        )

        # Narrow dictionary
        self.attach(FilterDict("narrowed-sources-dict"), "all-sources-dict", "narrowed-sources-list")

        # Get likelihoods for all token-positions for given sources
        self.attach_back(LikelihoodsForMultipleSources("source-likelihoods", self.chaining_embeddingBuilder_cfg))

        # Build high-likelihood chains
        self.attach(
            ChainingNode("all-chains", self.chaining_embeddingBuilder_cfg, use_bidirectional_chaining),
            "input-stripped",
            "narrowed-sources-list",
            "source-likelihoods",
        )

        # Filter overlapping chains out
        self.attach_back(FilterOverlappingChainsNode("filtered-chains"))

        # (MISC) Input text is split into tokens
        self.attach(
            TokenizeTextNode("input-tokenized", self.text_embeddingBuilder_cfg),
            "input-stripped",
            auxiliary=True,
        )

    @staticmethod
    def new(name: str, use_bidirectional_chaining: bool) -> SourceColoringPipeline:
        return SourceColoringPipeline(name, use_bidirectional_chaining)

    @staticmethod
    def new_extended(name: str = "coloring", use_bidirectional_chaining: bool = True) -> SourceColoringPipeline:
        pipeline = SourceColoringPipeline.new(name, use_bidirectional_chaining)

        # (MISC) Split source texts into tokens
        pipeline.attach(
            DictWrapperNode(TokenizeTextNode("narrowed-sources-dict-tokenized", pipeline.text_embeddingBuilder_cfg)),
            "narrowed-sources-dict",
            auxiliary=True,
        )

        # Add matched source texts to filtered chains
        pipeline.attach(
            AttachMetaData("chains-with-text"),
            "filtered-chains",
            "input-tokenized",
            "narrowed-sources-dict-tokenized",
            auxiliary=True,
        )

        return pipeline
