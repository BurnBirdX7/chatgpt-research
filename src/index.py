import os
from typing import Union, Tuple, List, Any, Dict

import faiss  # type: ignore
import pandas as pd  # type: ignore
import numpy as np

from . import EmbeddingsBuilder
from .pipeline import BaseDataDescriptor, base_data_descriptor, BaseNode
from .source_mapping import SourceMapping

__all__ = [
    'Index',
    'IndexDescriptor',
    'IndexFromSourcesNode'
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


class IndexDescriptor(BaseDataDescriptor[Index]):
    def store(self, data: Index) -> dict[str, base_data_descriptor.ValueType]:
        data.save()
        return {
            'index_file': os.path.abspath(data.config.index_file),
            'mapping_file': os.path.abspath(data.config.mapping_file),
            'threshold': data.config.threshold,
            'use_gpu': str(data.config.faiss_use_gpu),
        }

    def load(self, dic: dict[str, base_data_descriptor.ValueType]) -> Index:
        index_file = dic['index_file']
        mapping_file = dic['mapping_file']
        threshold = dic['threshold']
        use_gpu = dic['use_gpu']
        return Index.load(IndexConfig(
            index_file=index_file,
            mapping_file=mapping_file,
            threshold=threshold,
            faiss_use_gpu=bool(use_gpu),
        ))

    def get_data_type(self) -> type[Index]:
        return Index


class IndexFromSourcesNode(BaseNode):
    def __init__(self, name: str, embedding_builder_config: EmbeddingBuilderConfig):
        super().__init__(name,
                         [str],
                         IndexDescriptor())
        self.eb_config = embedding_builder_config

    def process(self, source_dict: Dict[str, str], *ignore) -> Index:
        """
        Accepts a dictionary (source_name -> source_text)
        """

        # Configuration of the index:
        output_folder = self.out_descriptor.artifacts_folder
        time_str = self.out_descriptor.get_timestamp_str()
        rand_str = self.out_descriptor.get_random_string(2)
        index_file_name = os.path.join(output_folder, f'{time_str}-{rand_str}.faiss_index')
        mapping_file_name = os.path.join(output_folder, f'{time_str}-{rand_str}.mapping.csv')

        index_cfg = IndexConfig(
            index_file=index_file_name,
            mapping_file=mapping_file_name,
        )

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



