from typing import Callable

import pytest

from src.pipeline.Block import Block
from src.pipeline.Blocks import map_block
from src.pipeline.Pipeline import Pipeline
from src.pipeline.DataDescriptors import *


@map_block(StrDescriptor(), int)
def Int2Str(inp: int) -> str:
    return str(inp)

@map_block(StrDescriptor(), str)
def ModifyString(inp: str) -> str:
    return inp + ".367"

@map_block(FloatDescriptor(), str)
def Str2Float(inp: str) -> float:
    return float(inp)

def test_simple_pipeline() -> None:
    pipeline = (Pipeline(Int2Str("i2s"))
                .attach(ModifyString("modStr"))
                .attach(Str2Float("s2f")))

    res, _ = pipeline.run(18)
    assert res == float(str(18) + '.367')

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

def test_empty_input_pipeline() -> None:
    pipeline = (
        Pipeline(ConstantBlock("constant", 42))
        .attach(IntMod("div-2", lambda x: x // 2))
        .attach(Int2Str("i2s"))
    )

    res, _ = pipeline.run(None)
    assert res == str(42 // 2)

@map_block(StrDescriptor())
def NoopBlock(inp: str) -> str:
    return inp

def test_diverging_paths():
    def str_merge(str1: str, str2: str) -> str:
        return f"{str1=} ... {str2=}"

    pipeline = (
        Pipeline(IntMod("mul2", lambda x: x * 2))
        .attach(IntMod("mul3", lambda x: x * 3))
        .attach(Int2Str("int2str-1"))
        .attach(IntMod("div2", lambda x: x // 2), "mul2")
        .attach(Int2Str("int2str-2"))
        .merge(NoopBlock[str, str]("output"), ["int2str-1", "int2str-2"], str_merge)
    )

    r"""
    mul2 -> mul3 -> int2str-1 -> | output |
    mul2 -> div2 -> int2str-2 -> | output |
    """

    val = 15
    res, _ = pipeline.run(val)

    path1 = str((val * 2) * 3)
    path2 = str((val * 2) // 2)
    s = str_merge(path1, path2)

    assert res == s

    print(f"{res=}")
