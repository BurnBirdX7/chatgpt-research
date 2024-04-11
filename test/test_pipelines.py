from typing import Callable

import pytest

from src.pipeline import BaseNode, mapping_node, Pipeline
from src.pipeline.data_descriptors import *


@mapping_node(StrDescriptor())
def Int2Str(inp: int) -> str:
    return str(inp)


@mapping_node(StrDescriptor())
def ModifyString(inp: str) -> str:
    return inp + ".367"


@mapping_node(FloatDescriptor())
def Str2Float(inp: str) -> float:
    return float(inp)


def test_simple_pipeline() -> None:
    pipeline = (
        Pipeline(Int2Str("i2s"))
        .attach_back(ModifyString("modStr"))
        .attach_back(Str2Float("s2f"))
    )

    res = pipeline.run(18).last_node_result
    assert res == float(str(18) + ".367")


class IntMod(BaseNode):

    def __init__(self, name: str, operation: Callable[[int], int]):
        super().__init__(name, [int], IntDescriptor())
        self.operation = operation

    def process(self, inp: int) -> int:
        return self.operation(inp)


class ConstantNode(BaseNode):
    def __init__(self, name: str, value: int):
        super().__init__(name, [type(None)], IntDescriptor())
        self.value = value

    def process(self, inp: None) -> int:
        return self.value


def test_empty_input_pipeline() -> None:
    pipeline = (
        Pipeline(ConstantNode("constant", 42))
        .attach_back(IntMod("div-2", lambda x: x // 2))
        .attach_back(Int2Str("i2s"))
    )

    res = pipeline.run().last_node_result
    assert res == str(42 // 2)


def test_diverging_paths():
    def concat(str1, str2):
        return f"{str1=} ... {str2=}"

    @mapping_node(StrDescriptor())
    def ConcatBlock(a: str, b: str) -> str:
        return concat(a, b)

    pipeline = (
        Pipeline(IntMod("mul2", lambda x: x * 2))
        .attach_back(IntMod("mul3", lambda x: x * 3))
        .attach_back(Int2Str("int2str-1"))
        .attach(IntMod("div2", lambda x: x // 2), "mul2")
        .attach_back(Int2Str("int2str-2"))
        .attach(ConcatBlock("output"), "int2str-1", "int2str-2")
    )

    r"""
    mul2 -> mul3 -> int2str-1 -> | output |
    mul2 -> div2 -> int2str-2 -> | output |
    """

    val = 15
    res = pipeline.run(val).last_node_result

    path1 = str((val * 2) * 3)
    path2 = str((val * 2) // 2)
    s = concat(path1, path2)

    assert res == s

    print(f"{res=}")
