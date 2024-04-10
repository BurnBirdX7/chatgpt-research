import itertools

from transformers import RobertaForMaskedLM

from src import QueryColbertServer
from src.chaining import ChainingNode, FilterChainsNode, Pos2ChainMapNode
from src.embeddings_builder import EmbeddingsFromTextNode, TokenizeTextNode, LikelihoodsForMultipleSources
from src.index import IndexFromSourcesNode, SearchIndexNode
from src.pipeline import Pipeline, mapping_node, DictDescriptor
from src.config import ColbertServerConfig, EmbeddingBuilderConfig
from src.text_processing import TextProcessingNode, remove_punctuation


@mapping_node(out_descriptor=DictDescriptor())
def FilterDict(dict_: dict, keys: list) -> dict:
    # Keys is a list of lists
    unique_keys = set(itertools.chain.from_iterable(keys))
    return {
        k: dict_[k]
        for k in unique_keys
    }


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

    # == Pipeline ==

    # First node strips input of punctuation
    pipeline = Pipeline(TextProcessingNode.new("input-stripped", remove_punctuation), name="text-coloring")

    # Node queries all sources that might contain similar text from ColBERT
    pipeline.attach_back(QueryColbertServer("all-sources-dict-raw", colbert_cfg))

    # Clear all retrieved texts of punctuation
    pipeline.attach_back(TextProcessingNode.new_for_dicts("all-sources-dict", remove_punctuation))

    # Generate embeddings from sources and build new FAISS index
    pipeline.attach_back(IndexFromSourcesNode("all-sources-index", text_eb_config))

    # (MISC) Input text is split into tokens
    pipeline.attach(TokenizeTextNode("input-tokenized", text_eb_config), "input-stripped")

    # Produce embeddings from input
    pipeline.attach(EmbeddingsFromTextNode("input-embeddings", text_eb_config), "input-stripped")

    # Get possible source names for every token-pos of the input text
    pipeline.attach(SearchIndexNode("narrowed-sources-list", k=10), "all-sources-index", "input-embeddings")

    # Narrow dictionary
    pipeline.attach(FilterDict("narrowed-sources-dict"), "all-sources-dict", "narrowed-sources-list")

    # Get likelihoods for all token-positions for given sources
    pipeline.attach_back(LikelihoodsForMultipleSources("source-likelihoods", chaining_eb_config))



    # Build high-likelihood chains
    pipeline.attach(
        ChainingNode("all-chains", chaining_eb_config, True),
        "input-stripped", "narrowed-sources", "source-likelihoods"
    )

    # Filter overlapping chains out
    pipeline.attach_back(FilterChainsNode("filtered-chains"))
    
    # Map every token-position onto corresponding chain
    pipeline.attach_back(Pos2ChainMapNode("token2chain"))

    return pipeline
