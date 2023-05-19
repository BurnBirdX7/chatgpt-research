from typing import Union

import faiss  # type: ignore
import pandas as pd  # type: ignore
import numpy as np

from .SourceMapping import SourceMapping
from .Roberta import Roberta
from .Embeddings import Embeddings
from .Config import Config

__all__ = ['Index']


class Index:
    def __init__(self, index: faiss.Index, mapping: SourceMapping, threshold: float = Config.threshold):
        self.index: faiss.Index = index
        self.mapping: SourceMapping = mapping
        self.threshold = threshold

    @staticmethod
    def load(index_file: str,
             mapping_file: str,
             threshold: float = Config.threshold,
             use_gpu: bool = Config.faiss_use_gpu) -> "Index":

        faiss_index = faiss.read_index(index_file)
        if use_gpu:
            faiss_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss_index)

        mapping = SourceMapping.read_csv(mapping_file)
        return Index(faiss_index, mapping, threshold)

    def save(self, index_file: str, mapping_file: str) -> None:
        faiss.write_index(faiss.index_gpu_to_cpu(self.index), index_file)
        self.mapping.to_csv(mapping_file)

    @staticmethod
    def from_embeddings(embeddings: Union[np.ndarray, pd.DataFrame],
                        mapping: SourceMapping,
                        threshold: float = Config.threshold,
                        use_gpu: bool = Config.faiss_use_gpu) -> "Index":
        """
        Builds index from provided embeddings
        :param embeddings: data to build the index
        :param threshold: threshold to divide data
        :param mapping: index to source mapping
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

        return Index(index, mapping, threshold)

    @staticmethod
    def from_wiki():
        embeddings = Embeddings(*Roberta.get_default())
        return Index.from_embeddings(*embeddings.from_wiki())
