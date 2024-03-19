
from .BaseConfig import BaseConfig, DefaultValue
from dataclasses import dataclass

@dataclass
class WikiServerConfig(BaseConfig):
    ip_address: str = DefaultValue('0.0.0.0')   # type: ignore
    port: int = DefaultValue(5678)              # type: ignore

    api_search_path = "/api/search"
    api_ping_path = "/api/ping"

    def get_api_url(self) -> str:
        return f'http://{self.ip_address}:{self.port}/api/search'
