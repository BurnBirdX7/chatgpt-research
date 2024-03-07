from dataclasses import dataclass

from .BaseConfig import BaseConfig, DefaultValue


@dataclass
class ServerConfig(BaseConfig):
    ip: str = DefaultValue("127.0.0.1")
    port: int = DefaultValue(4567)

    temp_dir: str = DefaultValue("./.temp/")
