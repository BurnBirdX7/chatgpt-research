"""
Arg: Text
Description
 * Find close sources in Source index (
"""
import sys
from typing import Dict

from search import Searcher
from src import Config, Roberta, EmbeddingsBuilder, Index


def main(args):
    if len(args) != 2:
        print("Supply text to analyze as first argument")

    text = args[1]

    with Searcher(Config.source_index_path, 20) as searcher:
        tokens = searcher.split_text(text)

        window_size = 3
        for i in range(len(tokens) - window_size + 1):
            window = tokens[i: i + window_size]
            searcher.add_clause(window)

        print("Searching...")
        sources = searcher.search()

    titles = list(map(lambda source: source.title, sources))
    bodies = map(lambda source: source.body, sources)
    source_dict = dict(zip(titles, bodies))
    print(source_dict)

    def source_provider(title: str) -> Dict[str, str]:
        return {title: source_dict[title]}

    builder = EmbeddingsBuilder(*Roberta.get_default(), normalize=True)
    embeddings, mapping = builder.from_sources(titles, source_provider)
    index = Index.from_embeddings(embeddings, mapping)
    index.save(Config.temp_index_file, Config.temp_mapping_file)


if __name__ == '__main__':
    main(sys.argv)
