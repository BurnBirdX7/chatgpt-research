from typing import List, Dict
from ..pipeline import BaseDataDescriptor, base_data_descriptor
from .chain import Chain


class ChainListDescriptor(BaseDataDescriptor):

    def store(self, data: List[Chain]) -> dict[str, base_data_descriptor.ValueType]:
        return {"chains": [chain.to_dict() for chain in data]}

    def load(self, dic: dict[str, base_data_descriptor.ValueType]) -> List[Chain]:
        return [
            ElasticChain.from_dict(d)  # type: ignore
            for d in dic["chains"]  # type: ignore
        ]

    def get_data_type(self) -> type[list]:
        return list


class Pos2ChainMappingDescriptor(BaseDataDescriptor[Dict[int, Chain]]):

    def store(self, data: Dict[int, Chain]) -> dict[str, base_data_descriptor.ValueType]:
        return {str(pos): chain.to_dict() for pos, chain in data.items()}

    def load(self, dic: dict[str, base_data_descriptor.ValueType]) -> Dict[int, Chain]:
        return {int(pos_str): ElasticChain.from_dict(chain_dict) for pos_str, chain_dict in dic.items()}  # type: ignore

    def get_data_type(self) -> type:
        return dict
