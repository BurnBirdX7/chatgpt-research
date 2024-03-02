# Creates an Index from wiki pages listed in `page_names` in config
# Input: config.page_names, online wiki
# Output: config.temp_index_file, config.temp_mapping_file

from scripts._elvis_data import elvis_related_articles
from src import Index, EmbeddingsBuilder, Wiki
from src.config.EmbeddingsConfig import EmbeddingsConfig
from src.config.IndexConfig import IndexConfig
from src.config.WikiConfig import WikiConfig


# TODO: Review
def build_index_from_wiki(wiki_config: WikiConfig, index_config: IndexConfig) -> None:
    builder = EmbeddingsBuilder(EmbeddingsConfig(normalize=True))
    sources = wiki_config.target_pages
    embeddings, mapping = builder.from_sources(sources, Wiki.parse)
    Index.from_embeddings(embeddings, mapping, index_config).save()


def build_index_from_elvis_articles():
    wiki_config = WikiConfig(elvis_related_articles)
    build_index_from_wiki(wiki_config, IndexConfig())


if __name__ == '__main__':
    build_index_from_elvis_articles()
