from dataclasses import dataclass, field

from .base_config import BaseConfig


@dataclass
class WebServerConfig(BaseConfig):
    ip: str = field(default="127.0.0.1")
    port: int = field(default=4567)
    temp_dir: str = field(default="./.temp/")
