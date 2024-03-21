from transformers import RobertaForMaskedLM

from src import QuerySourcesNode
from src.token_chain import ChainingNode
from src.embeddings_builder import EmbeddingsFromTextNode
from src.index import IndexFromSourcesNode, SearchIndexNode
from src.pipeline import Pipeline
from src.config import ColbertServerConfig, EmbeddingBuilderConfig


def get_coloring_pipeline() -> Pipeline:
    # Configs:
    colbert_cfg = ColbertServerConfig.load_from_env()
    text_eb_config = EmbeddingBuilderConfig()  # Default
    chaining_eb_config = EmbeddingBuilderConfig(
        model=RobertaForMaskedLM.from_pretrained("roberta-large")
    )

    # Pipeline:
    pipeline = Pipeline(QuerySourcesNode("possible-sources-dict", colbert_cfg))
    pipeline.attach_back(IndexFromSourcesNode("possible-sources-index", text_eb_config))
    pipeline.attach(EmbeddingsFromTextNode("input-embeddings", text_eb_config), "$input")
    pipeline.attach(SearchIndexNode("sources-list"), "possible-sources-index", "input-embeddings")
    pipeline.attach(
        ChainingNode("chains", chaining_eb_config),
        "$input", "sources-list", "possible-sources-dict"
    )

    return pipeline
