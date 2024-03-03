import pytest

from src.pipeline.Block import Block, InT, OutT
from src.pipeline.Blocks import map_block
from src.pipeline.DataDescriptors import *


def test_simple() -> None:
    @map_block(IntDescriptor(), int)
    def Mul2(a: int) -> int:
        return a * 2

    o = Mul2("mul2")
    res: float = o.process(10)
    assert res == 10 * 2
    assert o.__class__.__name__ == "Mul2"

def test_no_annotations_should_fail():

    with pytest.raises(TypeError):
        @map_block(EmptyDataDescriptor())
        def no_annotations(a):
            return None

    with pytest.raises(TypeError):
        @map_block(EmptyDataDescriptor())
        def no_return_annotation(a: int):
            return None

    with pytest.raises(TypeError):
        @map_block(IntDescriptor())
        def no_param_annotation(a) -> int:
            return a * 2


def test_wrong_descriptor_should_fail():
    with pytest.raises(TypeError):
        @map_block(IntDescriptor())
        def float_ret(a: int) -> float:
            return float(a)

    with pytest.raises(TypeError):
        @map_block(EmptyDataDescriptor())
        def int_ret(a: int) -> int:
            return a * 2

    with pytest.raises(TypeError):
        @map_block(IntDescriptor())
        def none_ret(_: int) -> None:
            return

def test_wrong_number_of_arguments_should_fail():
    with pytest.raises(TypeError):
        @map_block(FloatDescriptor())
        def float_ret(a: int, b: float) -> float:
            return float(a) + b


def test_block():
    class Handmade(Block[int, int]):
        def __init__(self, name: str):
            super().__init__(name, int, IntDescriptor())

        def process(self, inp: int) -> int:
            return inp * 2

    @map_block(IntDescriptor(), int)
    def DecoMade(inp: int) -> int:
        return inp * 2

    assert len(set(dir(DecoMade)).symmetric_difference(set(dir(Handmade)))) == 0
    assert Handmade.__name__ == "Handmade"
    assert DecoMade.__name__ == "DecoMade"



