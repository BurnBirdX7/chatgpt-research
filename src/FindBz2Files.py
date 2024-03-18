from __future__ import annotations

import os
import re

from src import WikiFile, ListWikiFileDescriptor
from src.pipeline import map_block


bz2_file_pattern = re.compile(r"^enwiki-(\d{8})-pages-articles-multistream(\d+).xml-p(\d+)p(\d+)\.bz2$")

@map_block(ListWikiFileDescriptor(), str)
def FindBz2Files(path: str) -> list[WikiFile]:
    dir_files = [
        file
        for file in os.listdir(path)
        if os.path.isfile(os.path.join(path, file))
    ]
    bz2_files = list[WikiFile]()

    for file in dir_files:
        match = bz2_file_pattern.match(file)
        if match is None:
            continue

        date, num, p_first, p_last = match.groups()
        bz2_files.append(WikiFile(
            os.path.join(path, file), date, int(num), int(p_first), int(p_last)
        ))

    bz2_files.sort()
    return bz2_files
