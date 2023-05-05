from typing import Union

import pandas as pd

from build_embeddings import build_embeddings
import roberta
import config
import faiss
import numpy as np


def build_index(embeddings: Union[np.ndarray, pd.DataFrame], use_gpu: bool = config.faiss_use_gpu) -> faiss.Index:
    """
    Builds index from provided embeddings
    :param embeddings: data to build the index
    :param use_gpu: if set, GPU is used to build the index
    :return: IndexFlatIP, or GpuIndexFlatIP id use_gpu is True
    """
    # C-contiguous order and np.float32 type are required
    if isinstance(embeddings, np.ndarray) and embeddings.flags['C_CONTIGUOUS']:
        data = embeddings.astype(np.float32)
    else:
        data = np.array(embeddings, order="C", dtype=np.float32)

    sequence_len, embedding_len = data.shape

    faiss.normalize_L2(data)
    print("Building index... ", end="")
    index = faiss.IndexFlatIP(embedding_len)
    if use_gpu:
        gpu_res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(gpu_res, 0, index)

    index.add(data)
    print("Done")
    return index


def build_index_from_file(file: str = config.embeddings_file) -> faiss.Index:
    print(f"Loading embeddings from '{file}'... ", end='')
    embeddings = pd.read_csv(file)
    print("Done")
    return build_index(embeddings)


def main() -> None:
    """
    Calculates embeddings, builds index and then saves it to file
    """
    e, r = build_embeddings(*roberta.get_default())
    r.to_csv(config.ranges_file)
    index = build_index(e, False)
    faiss.write_index(index, config.index_file)


if __name__ == '__main__':
    main()

