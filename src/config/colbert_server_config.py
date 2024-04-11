from .base_config import BaseConfig
from dataclasses import dataclass, field


@dataclass
class ColbertServerConfig(BaseConfig):
    # Dataclass fields
    ip_address: str = field(default="0.0.0.0")
    port: int = field(default=5678)

    # Class attributes
    api_search_path = "/api/search"
    api_ping_path = "/api/ping"

    def get_api_url(self) -> str:
        return f"http://{self.ip_address}:{self.port}/api/search"
