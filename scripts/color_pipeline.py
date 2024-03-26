from transformers import RobertaForMaskedLM

from src import QueryColbertServer
from src.token_chain import ChainingNode, FilterChainsNode, Pos2ChainMapNode
from src.embeddings_builder import EmbeddingsFromTextNode, TokenizeText
from src.index import IndexFromSourcesNode, SearchIndexNode
from src.pipeline import Pipeline
from src.config import ColbertServerConfig, EmbeddingBuilderConfig


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
    pipeline = Pipeline(QueryColbertServer("possible-sources-dict", colbert_cfg))
    pipeline.attach_back(IndexFromSourcesNode("possible-sources-index", text_eb_config))
    pipeline.attach(EmbeddingsFromTextNode("input-embeddings", text_eb_config), "$input")
    pipeline.attach(SearchIndexNode("sources-list"), "possible-sources-index", "input-embeddings")
    pipeline.attach(TokenizeText("input-tokenized", text_eb_config), "$input")
    pipeline.attach(
        ChainingNode("all-chains", chaining_eb_config),
        "$input", "sources-list", "possible-sources-dict"
    )
    
    pipeline.attach_back(FilterChainsNode("filtered-chains"))
    pipeline.attach_back(Pos2ChainMapNode("token2chain"))

    return pipeline
