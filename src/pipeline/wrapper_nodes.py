import logging

from .base_nodes import BaseNode, Node
from .data_descriptors import ComplexDictDescriptor, ComplexListDescriptor


__all__ = ['ListWrapperNode', 'DictWrapperNode']


class ListWrapperNode(BaseNode):
    def __init__(self, name: str, elem_node: Node):
        super().__init__(name, [list], ComplexListDescriptor(elem_node.out_descriptor))
        self.elem_node = elem_node
        self.elem_node.logger = logging.getLogger(f"{self.logger}.{self.elem_node.name}")

    @Node.logger.setter
    def logger(self, new_logger: logging.Logger):
        BaseNode.logger.fset(self, new_logger)
        self.elem_node.logger = logging.getLogger(f"{self.logger}.{self.elem_node.name}")

    def process(self, input_: list) -> list:
        return [
            self.elem_node.process(v)
            for v in input_
        ]


class DictWrapperNode(BaseNode):
    def __init__(self, name: str, elem_node: Node):
        super().__init__(name, [dict], ComplexDictDescriptor(elem_node.out_descriptor))
        self.elem_node = elem_node
        self.elem_node.logger = logging.getLogger(f"{self.logger}.{self.elem_node.name}")

    @Node.logger.setter
    def logger(self, new_logger: logging.Logger):
        BaseNode.logger.fset(self, new_logger)
        self.elem_node.logger = logging.getLogger(f"{self.logger}.{self.elem_node.name}")

    def process(self, input_: dict) -> dict:
        return {
            k: self.elem_node.process(v)
            for k, v in input_.items()
        }
