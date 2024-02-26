"""
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
from src.config.EmbeddingsConfig import EmbeddingsConfig
from src.config.IndexConfig import IndexConfig
from src.config.LuceneConfig import LuceneConfig


def build_index_from_potential_sources(text: str, index_config: IndexConfig) -> Index:
    lucene.getVMEnv().attachCurrentThread()
    with Searcher(LuceneConfig().index_path, 100) as searcher:
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

    builder = EmbeddingsBuilder(EmbeddingsConfig(normalize=True))
    embeddings, mapping = builder.from_sources(titles, source_provider)
    return Index.from_embeddings(embeddings, mapping, index_config)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Supply text to analyze as first argument")

    indexCfg = IndexConfig(
        index_file="__temp.index.csv",
        mapping_file="__temp.mapping.csv",
    )

    index = build_index_from_potential_sources(sys.argv[1], IndexConfig(
        index_file="__temp.index.csv",
        mapping_file="__temp.mapping.csv"
    ))

    index.save()
