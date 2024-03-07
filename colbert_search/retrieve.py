from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher


if __name__ == '__main__':
    with Run().context(RunConfig(nranks=1, experiment="test")):
        config = ColBERTConfig(
            root="../collections/experiments",
        )
        searcher = Searcher(index="test2", config=config)
        queries = Queries("../collections/queries.tsv")
        ranking = searcher.search_all(queries, k=100)
        ranking.save("rankings.tsv")
