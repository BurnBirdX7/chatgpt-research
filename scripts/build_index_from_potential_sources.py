"""
Script file provides useful functions to build index from potential sources

Run the script:
 Arg: Text
 Description
  * Uses PyLucene to find potential sources and builds Index from them
  * Index is saved into `temp_index_file` and `temp_mapping_file` specified in the Config
"""

import sys
from typing import Dict

import lucene
from search import Searcher
from src import EmbeddingsBuilder, Index
from src.config import EmbeddingBuilderConfig, IndexConfig, LuceneConfig


def build_index_from_potential_sources(text: str,
                                       index_config: IndexConfig,
                                       lucene_config: LuceneConfig,
                                       ) -> Index:
    """
    Builds Index from potential sources, Lucene Index is used to search for potential sources
    IndexConfig is used to generate new Index object, so only faiss_use_gpu parameter matters at the moment
    """

    lucene.getVMEnv().attachCurrentThread()
    with Searcher(lucene_config.index_path, 100) as searcher:
        tokens = searcher.split_text(text)

        print("Creating query")
        for window_size in range(1, 6):
            for i in range(len(tokens) - window_size + 1):
                window = tokens[i: i + window_size]
                searcher.add_clause(window)

        print("Searching...")
        sources = searcher.search()

    titles = list(map(lambda source: source.title, sources))
    bodies = map(lambda source: source.body, sources)
    source_dict = dict(zip(titles, bodies))

    def source_provider(title: str) -> Dict[str, str]:
        wikilink = f"https://en.wikipedia.org/wiki/{title}"
        return {wikilink: source_dict[title]}

    builder = EmbeddingsBuilder(EmbeddingBuilderConfig(normalize=True))
    embeddings, mapping = builder.from_sources(titles, source_provider)
    return Index.from_embeddings(embeddings, mapping, index_config)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Supply text to analyze as first argument")

    indexCfg = IndexConfig(
        index_file="__temp.index.csv",
        mapping_file="__temp.mapping.csv",
    )

    luceneCfg = LuceneConfig()

    index = build_index_from_potential_sources(
        sys.argv[1],
        indexCfg,
        luceneCfg
    )

    index.save()
