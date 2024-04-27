from collections import defaultdict
import typing as t

from src.chaining import Chain


class SingletonStorage:

    def __init__(self: "SingletonStorage"):
        self._registered_cached_funcs: t.List[t.Callable] = []

        # Dictionaries, [pipeline_key => contents]
        self.sources: t.DefaultDict[str, t.List[str]] = defaultdict(list)
        self.chains: t.DefaultDict[str, t.List[Chain]] = defaultdict(list)
        self.input_tokenized: t.Dict[str, t.List[str]] = {}

    def register_func(self, func: t.Callable):
        self._registered_cached_funcs.append(func)

    def clear_cache(self, keyset: t.Iterable[str]):
        for key in keyset:
            if key in self.sources:
                del self.sources[key]
            if key in self.chains:
                del self.chains[key]
            if key in self.input_tokenized:
                del self.input_tokenized[key]

        for func in self._registered_cached_funcs:
            func.cache_clear()  # type: ignore


storage: SingletonStorage = SingletonStorage()
