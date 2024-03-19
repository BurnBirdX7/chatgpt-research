from typing import Any

from src.pipeline import map_block
from src.pipeline.data_descriptors import DictDescriptor
from src.config import WikiServerConfig

import http.client

@map_block(DictDescriptor())
def QuerySources(cfg: WikiServerConfig) -> dict:
    pass

