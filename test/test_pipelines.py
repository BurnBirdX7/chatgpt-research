import pytest

from src.pipeline.Block import Block, InT, OutT
from src.pipeline.Pipeline import Pipeline
from src.pipeline.BasicDataDescriptors import *


class Int2Str(Block[int, str]):
    def __init__(self, name: str):
        super().__init__(name, IntDescriptor, StrDescriptor)

    def process(self, inp: int) -> str:
        return str(inp)


class ModifyString(Block[str, str]):
    def __init__(self, name: str):
        super().__init__(name, StrDescriptor, StrDescriptor)

    def process(self, inp: str) -> str:
        return inp + ".367"


class Str2Float(Block[str, float]):
    def __init__(self, name: str):
        super().__init__(name, StrDescriptor, FloatDescriptor)

    def process(self, inp: str) -> float:
        return float(inp)


def test_simple_pipeline():
    pipeline = Pipeline()
    pipeline = (pipeline
                .add_block(Int2Str("i2s"))
                .add_block(ModifyString("modStr"))
                .add_block(Str2Float("s2f")))

    assert 18.367 == pipeline.run(18)

