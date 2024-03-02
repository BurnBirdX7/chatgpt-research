from dataclasses import dataclass

from .BaseConfig import BaseConfig


@dataclass
class WikiConfig(BaseConfig):
    target_pages: list[str]
