from typing import Union

import pandas as pd

from build_embeddings import build_embeddings
import roberta
import config
import faiss
import numpy as np


def build_index(embeddings: Union[np.ndarray, pd.DataFrame], use_gpu: bool = config.faiss_use_gpu) -> faiss.Index:
    data = np.array(embeddings, order="C", dtype=np.float32)  # C-contiguous order and np.float32 type are required
    sequence_len, embedding_len = data.shape

    faiss.normalize_L2(data)
    print("Building index... ", end="")
    cpu_index = faiss.IndexFlatIP(embedding_len)
    if use_gpu:
        gpu_res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index)
    else:
        index = cpu_index

    index.add(data)
    print("Done")
    return index


def build_index_from_file(file: str = config.embeddings_file) -> faiss.Index:
    print("Loading embeddings... ", end='')
    embeddings = pd.read_csv(file)
    print("Done")
    return build_index(embeddings)


def main() -> None:
    e = build_embeddings(*roberta.get_default())
    index = build_index(e, False)
    faiss.write_index(index, config.index_file)


if __name__ == '__main__':
    main()

