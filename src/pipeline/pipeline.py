from __future__ import annotations

import datetime
import json
import os.path
from typing import Any, cast, Dict, Set, List

from src.config import BaseConfig
from src.pipeline.nodes import Node

PipelineHistory = Dict[str, str]


class PipelineError(RuntimeError):
    def __init__(self, message: str, history: dict[str, str]) -> None:
        super().__init__(message)
        self._history = history


class Pipeline:
    """
    Class that helps streamline data processing pipelines
    """

    def __init__(self: "Pipeline", inp: Node):
        self.input_node = inp
        self.output_node = inp
        self.artifacts_folder = "pipe-artifacts"
        self.in_types = inp.in_types
        self.nodes: Dict[str, Node] = {inp.name: inp}  # All Blocks in the pipeline
        self.execution_order = [inp]

        self.__must_cache_output: Set[str] = set()  # Set of blocks whose output should be cached
        self.__source_graph: Dict[str, List[str]] = {inp.name: ["$input"]}  # Edges point towards data source

    def attach_back(self, new_node: Node) -> "Pipeline":
        """
        Attaches block to the end of the pipeline
        ... -> [last_block] -> [new_block]
        """

        # Important checks
        if new_node.name in self.__source_graph:
            raise ValueError(f"Block with name '{new_node.name}' already exists in the pipeline")

        # Current output node of the pipeline will provide input for the new node
        input_node = self.output_node

        # Check type compatibility
        input_type: type = input_node.out_type
        if not new_node.is_input_type_acceptable([input_type]):
            raise TypeError(f"Pipeline's output type is not acceptable by the block that is being attached"
                            f"\tPipeline's output type is \"{input_type}\" and "
                            f"input type of the block \"{new_node.name}\" accepts \"{new_node.in_types}")

        if input_node is not self.output_node:
            self.__must_cache_output |= {input_node.name}

        # Set the node into place
        self.__set_node(new_node, [input_node])
        return self

    def attach(self,
               new_node: Node,
               *input_names: str) -> "Pipeline":
        r"""
        Attaches new node to outputs of nodes already existing in the pipeline

        pipeline.attach(new_node, "node1", "node2")

        ... -> [node1] --\
                         v
                     [new_node]
                         ^
        ... -> [node2] --/
        """

        # Important checks
        if new_node.name in self.__source_graph:
            raise ValueError(f"Block with name '{new_node.name}' already exists in the pipeline")

        inputs: List[Node] = []
        for name in input_names:
            if name not in self.nodes:
                raise ValueError(f"Block with name '{new_node.name}' does not exist in the pipeline")

            inp_block = self.nodes[name]
            inputs.append(inp_block)

        # Check type compatibility
        input_types: List[type] = [node.out_type for node in inputs]
        if not new_node.is_input_type_acceptable(input_types):
            raise TypeError(f"Node \"{new_node.name}\" expects types: {new_node.in_types}. "
                            f"Got types: {input_types}")

        # Set the node in place
        self.__set_node(new_node, inputs)
        self.__must_cache_output |= {name
                                     for name in input_names
                                     if name != self.output_node.name}

        return self

    def __set_node(self, new_node: Node, inputs: list[Node]):
        def get_name(b: Node) -> str:
            return b.name

        names = list(map(get_name, inputs))

        # Update graph and execution order
        self.execution_order.append(new_node)
        self.__source_graph[new_node.name] = names

        # General structure updated
        self.nodes[new_node.name] = new_node
        self.output_node = new_node
        new_node.set_artifacts_folder(self.artifacts_folder)

    def run(self, inp: Any = None) -> tuple[Any, PipelineHistory]:
        typs = [type(inp)]
        if not self.input_node.is_input_type_acceptable(typs):
            raise TypeError(f"Expected types {self.in_types} but got {typs}")

        history: Dict[str, str] = {"$input": inp}
        beginning_time = datetime.datetime.now()
        print(f"Starting pipeline [at {beginning_time}]...")

        try:
            return self.__run(inp, history, {"$input": inp}), history
        except Exception as e:
            raise PipelineError("Pipeline failed with an exception", history) from e
        finally:
            self.__save_history(beginning_time, history)

    def resume(self, history_file: str, block_name: str) -> Any:
        """
        Resume pipeline run from the block name, load all cachable data in memory
        """
        history_dir = os.path.dirname(history_file)
        history_dir = os.path.abspath(history_dir)
        with open(history_file, "r") as f:
            history = json.loads(f.read())

        # TODO: Add pipeline structure check
        # TODO: Add reachability check
        # TODO: Optimize excessive caching

        if block_name not in history:
            raise ValueError(f"f{block_name} isn't present in history file {history_file} [{history}]")

        beginning_time = datetime.datetime.now()
        print(f"Resuming pipeline [at {beginning_time}] [from {block_name}]...")

        cached_data: Dict[str, Any] = dict()

        # Load in cache all cachable outputs that are present in the history
        # and entry block itself
        for cached_block_name in self.__must_cache_output | {block_name}:
            if cached_block_name not in history:
                continue

            dic_filename = history[cached_block_name]
            dic_filename = os.path.join(history_dir, dic_filename)
            cached_data[cached_block_name] = self.__load_data(cached_block_name, dic_filename)

        if block_name in cached_data:
            inp = cached_data[block_name]
        else:
            dic_filename = history[block_name]
            dic_filename = os.path.join(history_dir, dic_filename)
            inp = self.__load_data(block_name, dic_filename)

        exec_order = self.execution_order
        idx = self.execution_order.index(self.nodes[block_name])
        self.execution_order = self.execution_order[idx+1:]

        try:
            return self.__run(inp, history, cached_data), history
        except Exception as e:
            raise PipelineError("Pipeline failed with an exception", history) from e
        finally:
            self.__save_history(beginning_time, history)
            self.execution_order = exec_order  # restore execution order

    def set_artifacts_folder(self, artifacts_folder: str):
        self.artifacts_folder = artifacts_folder
        for node in self.nodes.values():
            node.set_artifacts_folder(artifacts_folder)

    def __save_history(self, time: datetime.datetime, history: PipelineHistory):
        pipeline_history_file = f"pipeline_{Pipeline.format_time(time)}.json"
        pipeline_history_file = os.path.join(self.artifacts_folder, pipeline_history_file)
        print(f"Saving history [at {os.path.abspath(pipeline_history_file)}]")
        with open(pipeline_history_file, "w") as file:
            file.write(json.dumps(history))

    @staticmethod
    def get_timestamp_str() -> str:
        return Pipeline.format_time(datetime.datetime.now())

    @staticmethod
    def format_time(time: datetime.datetime) -> str:
        return time.strftime("%Y-%m-%d.%H-%M-%S")

    def __run(self, _input: Any, history: Dict[str, str], cached_data: Dict[str, Any]) -> Any:
        """
        Run the pipeline
        :param history: dictionary where key is the name of the block and the value is name of the file with essential information
        """

        if not os.path.exists(self.artifacts_folder):
            os.mkdir(self.artifacts_folder)

        prev_output: Any = _input
        prev_node_name: str = "$input"

        # Execution:
        for cur_node in self.execution_order:
            # Construct input for the current block
            source_names = self.__source_graph[cur_node.name]
            cur_input_data = []
            for s_name in source_names:
                cur_input_data.append(prev_output if s_name == prev_node_name else cached_data[s_name])

            # Type-check
            typs = [type(val) for val in cur_input_data]
            if not cur_node.is_input_type_acceptable(typs):
                raise TypeError(f"Input type(s) not acceptable by \"{cur_node.name}\" node\n"
                                f"expected types: {cur_node.in_types}, got: {typs}")

            # Process data
            prev_output = cur_node.process(*cur_input_data)
            prev_node_name = cur_node.name

            if cur_node.name in self.__must_cache_output:
                cached_data[cur_node.name] = prev_output

            # Save data to disk before resuming
            history[cur_node.name] = self.__store_data(cur_node, prev_output)

        return prev_output

    def __store_data(self, node: Node, data: Any) -> str:
        dic = node.out_descriptor.store(data)
        dic_name = f"pipe_{node.name}_{self.get_timestamp_str()}.json"
        filename = os.path.abspath(os.path.join(self.artifacts_folder, dic_name))
        with open(filename, "w") as file:
            dic_str = json.dumps(dic)
            file.write(dic_str)

        return dic_name

    def __load_data(self, block_name: str, dic_filename: str) -> Any:
        node = self.nodes[block_name]
        with open(dic_filename, "r") as file:
            dic_str = file.read()

        dic = json.loads(dic_str)
        return node.out_descriptor.load(dic)

