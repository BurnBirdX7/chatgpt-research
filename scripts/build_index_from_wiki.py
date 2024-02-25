# Creates an Index from wiki pages listed in `page_names` in config
# Input: config.page_names, online wiki
# Output: config.temp_index_file, config.temp_mapping_file

from src import Index, Config, EmbeddingsBuilder, Roberta, Wiki


def main() -> None:
    builder = EmbeddingsBuilder(*Roberta.get_default(), normalize=True)
    sources = Config.page_names
    embeddings, mapping = builder.from_sources(sources, Wiki.parse)
    index = Index.from_embeddings(embeddings, mapping)
    index.save(Config.temp_index_file, Config.temp_mapping_file)

if __name__ == '__main__':
    main()
