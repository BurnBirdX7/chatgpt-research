from typing import List

import numpy as np
import numpy.typing as npt
import matplotlib as mpl

from . import QueryColbertServerNode, Roberta
from .chaining.nodes import WideChaining, CollectTokenScoreNode
from .config import EmbeddingBuilderConfig, IndexConfig, ColbertServerConfig
from .embeddings_builder import EmbeddingsFromTextNode, LikelihoodsForMultipleSources, TokenizeTextNode
from .index import IndexFromSourcesNode, SearchIndexNode
from .pipeline import Pipeline, BaseNode, ListDescriptor
from .source_coloring_pipeline import FilterDict
from .text_processing import remove_punctuation, TextProcessingNode, remove_wiki_formatting


def normalize(slice_: npt.NDArray) -> npt.NDArray:
    min_ = np.min(slice_)
    max_ = np.max(slice_)

    return (slice_ - min_) / (max_ - min_)


class Score2ColorsNode(BaseNode):

    def __init__(self, name: str, cmap_name: str = "plasma"):
        super().__init__(name, [np.ndarray], ListDescriptor())
        self.cmap = mpl.colormaps[cmap_name]

    def process(self, scores: npt.NDArray[np.float32]) -> List[str]:
        norm = normalize(scores.reshape((-1, 1))).reshape(-1) * 0.7
        return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b, _ in self.cmap(norm, bytes=True)]


class ScoreColoringPipeline(Pipeline):

    def __init__(self, name):
        super().__init__(
            TextProcessingNode.new("input-stripped", remove_punctuation | remove_wiki_formatting), name=name
        )

        # CONFIGS:
        self.colbert_cfg: ColbertServerConfig = ColbertServerConfig.load_from_env()  # type: ignore
        self.text_embeddingBuilder_cfg = EmbeddingBuilderConfig(
            model=Roberta.get_default_masked_model(), normalize=True, centroid_file="artifacts/centroid-colbert.npy"
        )
        self.chaining_embeddingBuilder_cfg = EmbeddingBuilderConfig(
            model=Roberta.get_default_masked_model(),
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

        self.attach(
            WideChaining("wide-chains", self.chaining_embeddingBuilder_cfg),
            "input-stripped",
            "narrowed-sources-list",
            "source-likelihoods",
        )

        # (MISC) Input text is split into tokens
        self.attach(
            TokenizeTextNode("input-tokenized", self.text_embeddingBuilder_cfg),
            "input-stripped",
            auxiliary=True,
        )

        self.attach(CollectTokenScoreNode("scores"), "input-tokenized", "wide-chains")

        self.attach(Score2ColorsNode("colors"), "scores", auxiliary=True)
