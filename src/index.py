from typing import Union, Tuple, List

import faiss  # type: ignore
import pandas as pd  # type: ignore
import numpy as np

from .source_mapping import SourceMapping

__all__ = ['Index']

from .config.IndexConfig import IndexConfig


class Index:
    def __init__(self, index: faiss.Index, mapping: SourceMapping, config: IndexConfig):
        self.index: faiss.Index = index
        self.mapping: SourceMapping = mapping
        self.config = config

    @staticmethod
    def load(config: IndexConfig) -> "Index":
        """
        Loads index from disk
        """
        faiss_index = faiss.read_index(config.index_file)
        if config.faiss_use_gpu:
            faiss_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss_index)

        mapping = SourceMapping.read_csv(config.mapping_file)
        return Index(faiss_index, mapping, config)

    def save(self) -> None:
        faiss.write_index(faiss.index_gpu_to_cpu(self.index), self.config.index_file)
        self.mapping.to_csv(self.config.mapping_file)

    @staticmethod
    def from_embeddings(embeddings: Union[np.ndarray, pd.DataFrame],
                        mapping: SourceMapping,
                        config: IndexConfig) -> "Index":
        """
        Builds index from provided embeddings
        :param embeddings: data to build the index
        :param mapping: index to source mapping
        :param config: index configuration
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
        if config.faiss_use_gpu:
            gpu_res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(gpu_res, 0, index)

        index.add(data)
        print("Done")

        return Index(index, mapping, config)

    def dim(self):
        return self.index.d

    def search(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param x: 2D array with shape (N, dim)
        :return: tuple(indexes, distances)
        """
        dists, ids = self.index.search(x, 1)
        ids = np.squeeze(ids)
        dists = np.squeeze(dists)
        return ids, dists

    def get_source(self, idx: int):
        return self.mapping.get_source(idx)

    def get_embeddings_source(self, x: np.ndarray) -> Tuple[List[str], np.ndarray]:
        """
        Finds possible sources of embeddings

        :param x: 2D array with shape (N, dim)
        :return: tuple(source_strings, distances)
        """
        indexes, distances = self.search(x)
        return list(map(lambda i: self.get_source(i), indexes)), distances
