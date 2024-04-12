import os
import re

from .wiki_data_file import ListWikiFileDescriptor, WikiDataFile
from src.pipeline import mapping_node

__all__ = ["FindBz2Files"]

bz2_file_pattern = re.compile(r"^enwiki-(\d{8})-pages-articles-multistream(\d+).xml-p(\d+)p(\d+)\.bz2$")


@mapping_node(out_descriptor=ListWikiFileDescriptor())
def FindBz2Files(path: str) -> list:
    """
    Parameters
    ----------
        path : str
            Path to the folder that contains bz2 wiki archives.
            These archives must have names matchable by the ``bz2_file_pattern`` regex

    Notes
    -----
    ``FindBz2Files`` is a subclass of Node class, not a function

    See also
    --------
    mapping_node : Quick Node decorator
    """
    dir_files = [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
    bz2_files = list[WikiDataFile]()

    for file in dir_files:
        match = bz2_file_pattern.match(file)
        if match is None:
            continue

        date, num, p_first, p_last = match.groups()
        bz2_files.append(WikiDataFile(os.path.join(path, file), date, int(num), int(p_first), int(p_last)))

    bz2_files.sort()
    return bz2_files
