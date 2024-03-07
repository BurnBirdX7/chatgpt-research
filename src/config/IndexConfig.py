from dataclasses import dataclass

from .BaseConfig import BaseConfig, DefaultValue


@dataclass
class IndexConfig(BaseConfig):
    threshold: float = DefaultValue(0.8)                        # type: ignore
    faiss_use_gpu: bool = DefaultValue(False)                   # type: ignore
    index_file: str = DefaultValue("default.index.csv")         # type: ignore
    mapping_file: str = DefaultValue("default.mapping.csv")     # type: ignore