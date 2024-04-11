import logging
from abc import ABC

from . import BaseDataDescriptor
from .base_nodes import BaseNode, Node
from .data_descriptors import ComplexDictDescriptor, ComplexListDescriptor


__all__ = ["ListWrapperNode", "DictWrapperNode"]


class _WrapperNode(BaseNode, ABC):
    def __init__(
        self, elem_node: Node, in_type: type, out_descriptor: BaseDataDescriptor
    ):
        super().__init__(elem_node.name, [in_type], out_descriptor)
        self.elem_node = elem_node
        self.elem_node.logger = self.logger

    @Node.logger.setter
    def logger(self, new_logger: logging.Logger):
        BaseNode.logger.fset(self, new_logger)
        self.elem_node.logger = new_logger


class ListWrapperNode(_WrapperNode):
    """
    A wrapper node, that applies the inner node's process to every element of the incoming list and
    """

    def __init__(self, elem_node: Node):
        super().__init__(
            elem_node, list, ComplexListDescriptor(elem_node.out_descriptor)
        )

    def process(self, input_: list) -> list:
        return [self.elem_node.process(v) for v in input_]


class DictWrapperNode(_WrapperNode):
    def __init__(self, elem_node: Node):
        super().__init__(
            elem_node, dict, ComplexDictDescriptor(elem_node.out_descriptor)
        )

    def process(self, input_: dict) -> dict:
        return {k: self.elem_node.process(v) for k, v in input_.items()}
