from __future__ import annotations

import argparse
import sys

from src import FindBz2Files, Bz2Unpack
from colbert_search.prepare_wiki import PrepareWiki
from src.pipeline import Pipeline


def get_wiki_pipeline(output_path: str) -> Pipeline:
    """This pipeline collects bz2 wiki archives, unpacks them and converts into ColBERT-readable format

    Parameters
    ----------
    output_path : str
        Path to the folder where all output files will be dumped

    Returns
    -------
    Pipeline
        A pipeline that accepts path to the folder that contains one or more bz2 wiki archives

    Notes
    -----
    bz2 archives must have special naming, see details in FindBz2Files class' description
    """

    return (
        Pipeline(FindBz2Files("find-files"))
        .attach_back(Bz2Unpack("bz2-unpack", output_path))
        .attach_back(PrepareWiki("prepare-wiki", output_path))
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "python -m colbert_search wiki_pipeline",
        description="Takes directory that contains archived wiki dumps (.bz2 archives)"
        "and converts it into collection that can be supplied to ColBERT",
    )
    parser.add_argument("-i", dest="in_path", help="directory that contains archived wiki dumps", required=True)
    parser.add_argument("-o", dest="out_path", help="directory where artifacts will be stored", required=True)

    namespace = parser.parse_args()

    pipeline = get_wiki_pipeline(namespace.out_path)
    pipeline.run(namespace.in_path)
