from typing import Callable

import pytest

from src.pipeline.Block import Block, InT, OutT
from src.pipeline.Pipeline import Pipeline
from src.pipeline.BasicDataDescriptors import *


class Int2Str(Block[int, str]):
    def __init__(self, name: str):
        super().__init__(name, int, StrDescriptor())

    def process(self, inp: int) -> str:
        return str(inp)


class ModifyString(Block[str, str]):
    def __init__(self, name: str):
        super().__init__(name, str, StrDescriptor())

    def process(self, inp: str) -> str:
        return inp + ".367"


class Str2Float(Block[str, float]):
    def __init__(self, name: str):
        super().__init__(name, str, FloatDescriptor())

    def process(self, inp: str) -> float:
        return float(inp)


def test_simple_pipeline():
    pipeline = Pipeline()
    pipeline = (pipeline
                .add_block(Int2Str("i2s"))
                .add_block(ModifyString("modStr"))
                .add_block(Str2Float("s2f")))

    assert pipeline.run(18) == 18.367


class IntMod(Block[int, int]):

    def __init__(self, name: str, operation: Callable[[int], int]):
        super().__init__(name, int, IntDescriptor())
        self.operation = operation

    def process(self, inp: int) -> int:
        return self.operation(inp)


class ConstantBlock(Block[None, int]):
    def __init__(self, name: str, value: int):
        super().__init__(name, type(None), IntDescriptor())
        self.value = value

    def process(self, inp: None) -> int:
        return self.value


def test_empty_input_pipeline():
    pipeline = Pipeline()
    pipeline.add_block(ConstantBlock("constant", 42))
    pipeline.add_block(IntMod("div-2", lambda x: x // 2))
    pipeline.add_block(Int2Str("i2s"))

    assert pipeline.run() == str(42 // 2)


