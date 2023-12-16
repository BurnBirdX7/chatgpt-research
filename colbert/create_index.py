import os.path

from colbert.infra.config import RunConfig, ColBERTConfig
from colbert.infra.run import Run
from colbert import Indexer

if __name__ == '__main__':

    root = os.path.dirname(os.path.realpath(__file__))
    collections_dir = os.path.join(root, "collections")
    checkpoint_dir = os.path.join(root, "checkpoint")
    experiment_dir = os.path.join(collections_dir, "collections/experiments")

    collection_file = os.path.join(collections_dir, "long-entries.tsv")
    index_name = "test_long_entries"

    with Run().context(RunConfig(nranks=1, experiment="wiki", root=root)):
        indexer = Indexer(checkpoint=checkpoint_dir)
        indexer.index(name=index_name, collection=collection_file, overwrite=True)
