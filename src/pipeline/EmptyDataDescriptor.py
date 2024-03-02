from src.pipeline.BaseDataDescriptor import BaseDataDescriptor, T


class EmptyDataDescriptor(BaseDataDescriptor[None]):
    @classmethod
    def get_data_type(cls) -> type[None]:
        return type(None)

    def store(self, data: None) -> dict[str, str]:
        return {}

    def load(self, dic: dict[str, str]) -> None:
        return None
