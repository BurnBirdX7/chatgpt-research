import pytest

from src.pipeline.nodes import BaseNode
from src.pipeline.mapping_node import mapping_node
from src.pipeline.data_descriptors import *


def test_simple() -> None:
    @mapping_node(IntDescriptor())
    def Mul2(a: int) -> int:
        return a * 2

    o = Mul2("mul2")
    res: float = o.process(10)
    assert res == 10 * 2
    assert o.__class__.__name__ == "Mul2"

def test_no_annotations_should_fail():

    with pytest.raises(TypeError):
        @mapping_node(EmptyDataDescriptor())
        def no_annotations(a):
            return None

    with pytest.raises(TypeError):
        @mapping_node(EmptyDataDescriptor())
        def no_return_annotation(a: int):
            return None

    with pytest.raises(TypeError):
        @mapping_node(IntDescriptor())
        def no_param_annotation(a) -> int:
            return a * 2


def test_wrong_descriptor_should_fail():
    with pytest.raises(TypeError):
        @mapping_node(IntDescriptor())
        def float_ret(a: int) -> float:
            return float(a)

    with pytest.raises(TypeError):
        @mapping_node(EmptyDataDescriptor())
        def int_ret(a: int) -> int:
            return a * 2

    with pytest.raises(TypeError):
        @mapping_node(IntDescriptor())
        def none_ret(_: int) -> None:
            return


def test_node():
    class Handmade(BaseNode):
        def __init__(self, name: str):
            super().__init__(name, [int], IntDescriptor())

        def process(self, inp: int) -> int:
            return inp * 2

    @mapping_node(IntDescriptor())
    def DecoMade(inp: int) -> int:
        return inp * 2

    assert len(set(dir(DecoMade)).symmetric_difference(set(dir(Handmade)))) == 0
    assert Handmade.__name__ == "Handmade"
    assert DecoMade.__name__ == "DecoMade"
    assert Handmade('a').process(5) == DecoMade('b').process(5)

def test_node2():

    class Handmade(BaseNode):
        def __init__(self, name: str):
            super().__init__(name, [int], IntDescriptor())

        def process(self, a: int, b: int) -> int:
            return a + b

    @mapping_node(IntDescriptor())
    def DecoMade(x: int, y: int) -> int:
        return x + y

    assert len(set(dir(DecoMade)).symmetric_difference(set(dir(Handmade)))) == 0
    assert Handmade.__name__ == "Handmade"
    assert DecoMade.__name__ == "DecoMade"
    assert Handmade('a').process(5, 10) == DecoMade('b').process(5, 10)

