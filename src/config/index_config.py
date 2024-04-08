from dataclasses import dataclass, field
from .base_config import BaseConfig


@dataclass
class IndexConfig(BaseConfig):
    threshold: float = field(default=0.8)
    faiss_use_gpu: bool = field(default=False)
    index_file: str = field(default="default.faiss_index")
    mapping_file: str = field(default="default.mapping.csv")
