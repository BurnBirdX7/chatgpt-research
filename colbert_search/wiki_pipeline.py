from __future__ import annotations

import sys

from colbert_search.FindBz2Files import FindBz2Files
from colbert_search.prepare_wiki import PrepareWiki
from colbert_search.Bz2Unpack import Bz2Unpack
from src.pipeline import Pipeline


def get_wiki_pipeline(output_path: str) -> Pipeline:
    return (
        Pipeline(FindBz2Files("find-files"))
        .attach(Bz2Unpack("bz2-unpack", output_path))
        .attach(PrepareWiki("prepare-wiki", output_path))
    )


if __name__ == '__main__':
    in_path = sys.argv[1]
    out_path = sys.argv[2]

    pipeline = get_wiki_pipeline(out_path)
    pipeline.run(in_path)



