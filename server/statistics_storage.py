from collections import defaultdict
from typing import List, Callable, Dict

from src.chaining import Chain


class SingletonStorage:

    def __init__(self):
        self._registered_cached_funcs: List[Callable] = []

        # Dictionaries, [pipeline_key => contents]
        self.sources: Dict[str, List[str]] = defaultdict(list)
        self.chains: Dict[str, List[Chain]] = defaultdict(list)
        self.input_tokenized: List[str] = []

    def register_func(self, func: Callable):
        self._registered_cached_funcs.append(func)

    def clear_cache(self):
        self.sources = defaultdict(list)
        self.chains = defaultdict(list)
        self.input_tokenized = []
        for func in self._registered_cached_funcs:
            func.cache_clear()  # type: ignore


storage: SingletonStorage = SingletonStorage()
