
import json
import os
import re
import shutil
import sys
from typing import Tuple, List

from colbert_search.WikiFile import WikiFile
from src import SourceMapping

def find_matches(wikidict_pipe_path: str, directory: str) -> List[Tuple[WikiFile, str, int, bool]]:
    dic = json.load(open(wikidict_pipe_path, 'r'))['list']

    files = []
    for d, _, filesnames in os.walk(directory):
        files += [os.path.join(d, f) for f in filesnames]

    lst: List[Tuple[WikiFile, str, int, bool]] = []
    for d in dic:
        w = WikiFile.from_dict(d)
        print("file:", w.path)

        pattern_str = fr"^wiki-{w.num}-{w.p_first}_(\w+)_(\d+).tsv$"
        print(f" ~~~ pattern:", pattern_str)

        pattern = re.compile(pattern_str)
        for filename in files:
            m = re.match(pattern, os.path.basename(filename))
            if m:
                is_source = m.group(1) == 'sources'
                partition = int(m.group(2))
                print(f" > found: {filename}, is_source: {is_source}")
                lst.append((w, filename, partition, is_source))

    return lst

def tsv_table_to_csv_mapping(tsv_filename: str) -> SourceMapping:
    mapping = SourceMapping()
    with open(tsv_filename, "r") as file:
        for i, line in enumerate(file):
            parts = line.split('\t', 1)
            if len(parts) != 2:
                print(f"Got wrong line format, line #{i}:")
                continue
            id, url = parts
            clean_url = url.replace('\t', '_').replace(' ', '_').replace('\'', '')
            mapping.append_interval(1, clean_url)

    return mapping


def update_formats(wikidict_pipe_path, directory: str, out_directory: str) -> None:
    matches = find_matches(wikidict_pipe_path, directory)

    for wikiFile, filepath, partition, is_source in matches:

        print(f" -> {filepath}")

        name = f'wiki-{wikiFile.num}-p{wikiFile.p_first}-p{wikiFile.p_last}'
        if is_source:
            print(" -- remapping")
            mapping = tsv_table_to_csv_mapping(filepath)
            name += f'_source_{partition}.csv'
            path = os.path.join(out_directory, name)
            mapping.to_csv(path)
        else:
            print(" -- copying")
            name += f'_passages_{partition}.tsv'
            path = os.path.join(out_directory, name)
            shutil.copy2(filepath, path)

        print(f" <- {path}")


if __name__ == '__main__':
    """
    Arguments:
    wikidict_pipe_path: Path to pipe output produced by ListWikiFileDescriptor
    directory: where to find produced files
    out_directory: where to put reprocessed files
    """
    update_formats(sys.argv[1], sys.argv[2], sys.argv[3])
