from __future__ import annotations

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


if __name__ == '__main__':
    in_path = sys.argv[1]
    out_path = sys.argv[2]

    pipeline = get_wiki_pipeline(out_path)
    pipeline.run(in_path)



