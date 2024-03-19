from dataclasses import dataclass

from .base_config import BaseConfig, DefaultValue


@dataclass
class WebServerConfig(BaseConfig):
    ip: str = DefaultValue("127.0.0.1")       # type: ignore
    port: int = DefaultValue(4567)            # type: ignore
    temp_dir: str = DefaultValue("./.temp/")  # type: ignore
