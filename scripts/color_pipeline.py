from transformers import RobertaForMaskedLM

from src import QueryColbertServer
from src.chaining import ChainingNode, FilterChainsNode, Pos2ChainMapNode
from src.embeddings_builder import EmbeddingsFromTextNode, TokenizeTextNode, EmbeddingsForMultipleSources
from src.index import IndexFromSourcesNode, SearchIndexNode
from src.pipeline import Pipeline
from src.config import ColbertServerConfig, EmbeddingBuilderConfig
from src.text_processing import TextProcessingNode, remove_punctuation


def get_coloring_pipeline() -> Pipeline:
    # Configs:
    colbert_cfg = ColbertServerConfig.load_from_env()
    text_eb_config = EmbeddingBuilderConfig(
        normalize=True,
        centroid_file='artifacts/centroid-colbert.npy'
    )  # Default
    chaining_eb_config = EmbeddingBuilderConfig(
        model=RobertaForMaskedLM.from_pretrained("roberta-large").to(text_eb_config.model.device),
        centroid_file='artifacts/centroid-colbert.npy'
    )

    # Pipeline:
    pipeline = Pipeline(TextProcessingNode.new("strip-punctuation", remove_punctuation))
    pipeline.attach_back(QueryColbertServer("possible-sources-dict-raw", colbert_cfg))
    pipeline.attach_back(TextProcessingNode.new_for_dicts("possible-sources-dict", remove_punctuation))
    pipeline.attach_back(IndexFromSourcesNode("possible-sources-index", text_eb_config))

    pipeline.attach(TokenizeTextNode("input-tokenized", text_eb_config), "strip-punctuation")
    pipeline.attach(EmbeddingsFromTextNode("input-embeddings", text_eb_config), "strip-punctuation")
    pipeline.attach(SearchIndexNode("sources-names", k=10), "possible-sources-index", "input-embeddings")
    pipeline.attach(
        EmbeddingsForMultipleSources("source-embeddings", chaining_eb_config),
        "sources-names", "possible-sources-dict")
    pipeline.attach(
        ChainingNode("all-chains", chaining_eb_config),
        "strip-punctuation", "sources-names", "source-embeddings"
    )
    
    pipeline.attach_back(FilterChainsNode("filtered-chains"))
    pipeline.attach_back(Pos2ChainMapNode("token2chain"))

    return pipeline
