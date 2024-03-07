from __future__ import annotations

import bz2
import os.path
import sys

from colbert_search.FindBz2Files import FindBz2Files
from colbert_search.prepare_wiki import prepare_wiki

def wiki_pipeline(in_path: str, out_path: str):
    lst = FindBz2Files("--").process(in_path)
    lst = lst[:1]

    print(f"Found {len(lst)} files")
    for file in lst:
        print(f"\t - {file.path}")

    for file in lst:
        print(f"Unpacking {file.path}...", end="", flush=True)
        unpacked_filepath = f"wiki-{file.num}-{file.p_first}.xml"
        unpacked_filepath = os.path.join(out_path, unpacked_filepath)
        with open(file.path, "rb") as packed_f, open(unpacked_filepath, "wb") as unpacked_f:
            unpacked_f.write(bz2.decompress(packed_f.read()))
        print(f"done => {unpacked_filepath}")
        file.path = unpacked_filepath

    for file in lst:
        prepare_wiki(f"wiki-{file.num}-{file.p_first}", file.path, out_path)

if __name__ == '__main__':
    wiki_pipeline(sys.argv[1], sys.argv[2])




