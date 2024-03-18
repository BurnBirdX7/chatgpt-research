import bz2
import os
from typing import List

from colbert_search.FindBz2Files import WikiFile, ListWikiFileDescriptor
from src.pipeline import Block


class Bz2Unpack(Block[List[WikiFile], List[WikiFile]]):

    def __init__(self, name: str, output_dir: str):
        super().__init__(name, list, ListWikiFileDescriptor())
        self.output_dir = output_dir

    def process(self, lst: List[WikiFile]) -> List[WikiFile]:
        for file in lst:
            print(f"Unpacking {file.path}...", end="", flush=True)
            unpacked_filepath = f"wiki-{file.num}-p{file.p_first}-p{file.p_last}.xml"
            unpacked_filepath = os.path.join(self.output_dir, unpacked_filepath)
            with open(file.path, "rb") as packed_f, open(unpacked_filepath, "wb") as unpacked_f:
                unpacked_f.write(bz2.decompress(packed_f.read()))
            print(f"done => {unpacked_filepath}")
            file.path = unpacked_filepath

        return lst
