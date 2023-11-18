from colbert.infra.config import RunConfig, ColBERTConfig
from colbert.infra.run import Run
from colbert import Indexer

if __name__ == '__main__':
    with Run().context(RunConfig(nranks=1, experiment="test")):

        config = ColBERTConfig(
            nbits=2,
            root="../collections/experiments/",
        )
        indexer = Indexer(checkpoint="../checkpoint", config=config)
        indexer.index(name="test2", collection="../collections/fever.tsv", overwrite=True)
