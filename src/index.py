from __future__ import annotations

import os
from typing import Union, Tuple, List, Any, Dict

import faiss  # type: ignore
import pandas as pd  # type: ignore
import numpy as np

from . import EmbeddingsBuilder
from .pipeline import BaseDataDescriptor, base_data_descriptor, BaseNode, ListDescriptor
from .source_mapping import SourceMapping

__all__ = [
    'Index',
    'IndexDescriptor',
    'IndexFromSourcesNode',
    'SearchIndexNode'
]

from .config import IndexConfig, EmbeddingBuilderConfig


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
        Search for similar vectors (top-1)

        :param x: 2D array with shape (N, dim), where N is vector count and dim is vector length
        :return: tuple(indexes, distances)
        """
        dists, ids = self.index.search(x, 1)
        ids = np.squeeze(ids)
        dists = np.squeeze(dists)
        return ids, dists

    def search_topk(self, x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Same as search, but for top-k similar vectors

        :param x: 2D array with shape (N, dim)
        :return: tuple(indexes, distances) where
                    indexes and distances are numpy array of shape (N, k)
        """
        dists, ids = self.index.search(x, k)
        return ids, dists

    def get_source(self, idx: int) -> str:
        return self.mapping.get_source(idx)

    def get_sources(self, indexes: List[int] | np.ndarray) -> list[str]:
        return list(map(lambda x: self.get_source(x), indexes))

    def get_embeddings_source(self, x: np.ndarray) -> Tuple[List[str], np.ndarray]:
        """
        Finds possible sources of embeddings

        :param x: 2D array with shape (N, dim)
        :return: tuple(source_strings, distances)
        """
        indexes, distances = self.search(x)
        return list(map(lambda i: self.get_source(i), indexes)), distances

    def get_topk_sources_for_embeddings(self, x: np.ndarray, k: int) -> List[List[str]]:
        """
        Finds top-k most probable sources for the embeddings

        :param x: 2D array with shape (N, dim)
        :param k: k in the top-k
        :return: list of lists with dimensions (N, k)
        """

        indexes, _ = self.search_topk(x, k)
        return [
            self.get_sources(idxs)
            for idxs in indexes  # indexes has (N, k) dims => idxs has (k,) dims
        ]


class IndexDescriptor(BaseDataDescriptor[Index]):
    def store(self, data: Index) -> Dict[str, base_data_descriptor.ValueType]:

        # Generate names for files:
        output_folder = self.artifacts_folder
        time_str = self.get_timestamp_str()
        rand_str = self.get_random_string(2)
        index_file_name = os.path.join(output_folder, f'{time_str}-{rand_str}.faiss_index')
        mapping_file_name = os.path.join(output_folder, f'{time_str}-{rand_str}.mapping.csv')

        # Replace current config names with the generated
        data.config.index_file = index_file_name
        data.config.mapping_file = mapping_file_name

        # Save
        data.save()

        # Return data
        return {
            'index_file': os.path.abspath(index_file_name),
            'mapping_file': os.path.abspath(mapping_file_name),
            'threshold': data.config.threshold,
            'use_gpu': str(data.config.faiss_use_gpu),
        }

    def load(self, dic: Dict[str, base_data_descriptor.ValueType]) -> Index:
        index_file = dic['index_file']
        mapping_file = dic['mapping_file']
        threshold = dic['threshold']
        use_gpu = dic['use_gpu']
        return Index.load(IndexConfig(
            index_file=str(index_file),
            mapping_file=str(mapping_file),
            threshold=float(threshold), # type: ignore
            faiss_use_gpu=bool(use_gpu),
        ))

    def get_data_type(self) -> type:
        return Index


class IndexFromSourcesNode(BaseNode):
    """
    Node processes dictionary (source_name -> source_text) into Index that can be searched
    """

    def __init__(self, name: str, embedding_builder_config: EmbeddingBuilderConfig):
        super().__init__(name,
                         [dict],
                         IndexDescriptor())
        self.eb_config = embedding_builder_config

    def process(self, source_dict: Dict[str, str], *ignore) -> Index:
        """
        Accepts a dictionary (source_name -> source_text)
        """
        index_cfg = IndexConfig()

        # Build embeddings
        eb = EmbeddingsBuilder(self.eb_config)
        all_embeddings = None
        mapping = SourceMapping()

        for source_name, source_text in source_dict.items():
            embeddings = eb.from_text(source_text)
            mapping.append_interval(len(embeddings), source_name)

            if all_embeddings is None:
                all_embeddings = embeddings
            else:
                all_embeddings = np.concatenate([all_embeddings, embeddings])

        return Index.from_embeddings(all_embeddings, mapping, index_cfg)


class SearchIndexNode(BaseNode):
    def __init__(self, name: str, k: int = 10):
        super().__init__(name, [Index, np.ndarray], ListDescriptor())
        self.k = k

    def process(self, index: Index, embeddings: np.ndarray) -> List[List[str]]:
        """
        Returns a list of lists of sources for given embeddings,
        top-k probable sources per embeddings
        """
        return index.get_topk_sources_for_embeddings(embeddings, self.k)
