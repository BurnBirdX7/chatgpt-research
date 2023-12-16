import os.path
import sys

from colbert.infra.config import RunConfig
from colbert.infra.run import Run
from colbert import Indexer


def change_root():
    pass


def create_index(collection_name: str):
    root = os.path.dirname(os.path.realpath(__file__))
    collections_dir = os.path.join(root, "collections")
    checkpoint_dir = os.path.join(root, "checkpoint")

    collection_file = os.path.join(collections_dir, f"{collection_name}.tsv")

    with Run().context(RunConfig(nranks=1, experiment="wiki", root=root)):
        indexer = Indexer(checkpoint=checkpoint_dir)
        indexer.index(name=collection_name, collection=collection_file, overwrite=True)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python -m colbert create_index <collection_name>")
        print("Example: python -m colbert create_index fever")
        print("          collection will be loaded from collections/fever.tsv")
        exit(1)

    change_root()
    create_index(sys.argv[1])
