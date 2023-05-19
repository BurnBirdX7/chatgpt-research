import faiss
from src.classes import SourceMapping

__all__ = ['Index']


class Index:
    def __init__(self, index: faiss.Index, mapping: SourceMapping, threshold: float):
        self.index: faiss.Index = index
        self.mapping: SourceMapping = mapping
        self.threshold = threshold

    @staticmethod
    def load(index_file: str, mapping_file: str, threshold: float, use_gpu: bool = True) -> "Index":
        faiss_index = faiss.read_index(index_file)
        if use_gpu:
            faiss_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, faiss_index)

        mapping = SourceMapping.read_csv(mapping_file)
        return Index(faiss_index, mapping, threshold)

    def save(self, index_file: str, mapping_file: str) -> None:
        faiss.write_index(faiss.index_gpu_to_cpu(self.index), index_file)
        self.mapping.to_csv(mapping_file)
