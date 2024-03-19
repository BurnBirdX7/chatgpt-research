

from src import QuerySources
from src.pipeline import Pipeline
from src.config import ColbertServerConfig


def get_coloring_pipeline() -> Pipeline:
    # Configs:
    colbert_cfg = ColbertServerConfig.load_from_env()

    # Pipeline:
    pipeline = Pipeline(QuerySources("query", colbert_cfg))

    return pipeline

