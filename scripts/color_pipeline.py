

from src import QuerySources
from src.index import IndexFromSourcesNode
from src.pipeline import Pipeline
from src.config import ColbertServerConfig, EmbeddingBuilderConfig


def get_coloring_pipeline() -> Pipeline:
    # Configs:
    colbert_cfg = ColbertServerConfig.load_from_env()
    text_eb_config = EmbeddingBuilderConfig()  # Default

    # Pipeline:
    pipeline = Pipeline(QuerySources("query-sources", colbert_cfg))
    pipeline.attach_back(IndexFromSourcesNode("index-sources", text_eb_config))

    return pipeline

