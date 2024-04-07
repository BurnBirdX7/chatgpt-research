from __future__ import annotations

import datetime
import json
import logging
import os.path
import time
from typing import Any, Dict, Set, List, Tuple

from src.pipeline.nodes import Node
from .pipeline_result import PipelineResult, PipelineHistory, NodeStatistics


class PipelineError(RuntimeError):
    def __init__(self, message: str, history: dict[str, str]) -> None:
        super().__init__(message)
        self._history = history


class Pipeline:
    """
    Class that helps streamline data processing pipelines
    """

    def __init__(self: "Pipeline",
                 inp: Node,
                 store_intermediate_data: bool = True,
                 store_optional_data: bool = False,
                 name: str = "pipeline") -> None:
        """
        Create a Pipeline object

        Parameters
        ----------
        inp : Node
            The input node, this node accepts input that passed to the pipeline
            This node can have one or zero inputs
            Other nodes also can acquire input value by receiving "$input" input

        store_intermediate_data : bool, default=True
            Whether to store outputs of the nodes to disk

        store_optional_data : bool, default=False
            Whether to store optional data, ignored if store_intermediate_data is False

        name : str, default="pipeline"
            Name of the pipeline, used for logging


        Raises
        ------
        ValueError
            If inp accepts more than one input

        """

        # Types:
        if len(inp.in_types) > 1:
            raise ValueError("First node must accept zero or one parameter")
        elif len(inp.in_types) == 1:
            self.in_type = inp.in_types[0]
        else:
            self.in_type = type(None)

        # Options:
        self.store_intermediate_data = store_intermediate_data
        self.store_optional_data = store_optional_data

        # Pipeline setup
        self.input_node = inp
        self.output_node = inp
        self.artifacts_folder = "pipe-artifacts"
        self.nodes: Dict[str, Node] = {inp.name: inp}  # All nodes in the pipeline
        self.execution_plan = [inp]
        self.graph: Dict[str, List[str]] = {inp.name: ["$input"]}  # Edges point towards data source
        self.__must_cache_output: Set[str] = set()  # Set of nodes whose output should be cached

        # Misc
        self.name = name
        self.logger = logging.getLogger(f"pipeline.{self.name}")

    def attach_back(self, new_node: Node) -> "Pipeline":
        """
        Attaches node to the end of the pipeline
        ... -> [last_node] -> [new_node]
        """

        # Important checks
        if new_node.name in self.graph:
            raise ValueError(f"Node with name '{new_node.name}' already exists in the pipeline")

        # Current output node of the pipeline will provide input for the new node
        input_node = self.output_node

        # Check type compatibility
        input_type: type = input_node.out_type
        if not new_node.is_input_type_acceptable([input_type]):
            raise TypeError(f"Pipeline's output type is not acceptable by the node that is being attached"
                            f"\tPipeline's output type is \"{input_type}\" and "
                            f"input type of the node \"{new_node.name}\" accepts \"{new_node.in_types}")

        if input_node is not self.output_node:
            self.__must_cache_output |= {input_node.name}

        # Set the node into place
        self.__set_node(new_node, [input_node.name])
        return self

    def attach(self,
               new_node: Node,
               *input_names: str) -> "Pipeline":
        r"""
        Attaches new node to outputs of nodes already existing in the pipeline

        Examples
        --------
        pipeline.attach(new_node, "node1", "node2")

         > Makes `new_node` a pipeline's output
         > Makes outputs of blocks `node1` and `node2` inputs for the new node

        Parameters
        ----------
        new_node : Node
            A node to attach

        *input_names : str
            The names of the inputs for the new nodes
            Number of names should be equal the number of inputs of the node
            Types of outputs should be compatible with inputs of the node

        """

        # Important checks
        if new_node.name in self.graph:
            raise ValueError(f"Node with name '{new_node.name}' already exists in the pipeline")

        input_types: List[type] = []
        for name in input_names:
            if name == "$input":
                input_types.append(self.in_type)
            elif name in self.nodes:
                input_types.append(self.nodes[name].out_type)
            else:
                raise ValueError(f"Node with name '{name}' does not exist in the pipeline")

        # Check type compatibility
        if not new_node.is_input_type_acceptable(input_types):
            raise TypeError(f"Node \"{new_node.name}\" expects types: {new_node.in_types}. "
                            f"Got types: {input_types}")

        # Set the node in place
        self.__set_node(new_node, list(input_names))
        self.__must_cache_output |= {name
                                     for name in input_names
                                     if name != self.output_node.name}

        return self

    def force_caching(self, node_name: str) -> None:
        """
        Force caching of node's output during pipeline execution
        """
        if node_name in self.nodes:
            self.__must_cache_output.add(node_name)

    def check_prerequisites(self):
        """Throws ValueError if prerequisites are not met"""
        for node in self.execution_plan:
            r = node.prerequisite_check()
            if r is not None:
                raise ValueError(f"Prerequisite check failed for node \"{node.name}\": {r}")


    def __set_node(self, new_node: Node, input_names: list[str]):
        # Update graph and execution order
        self.execution_plan.append(new_node)
        self.graph[new_node.name] = input_names

        # General structure updated
        self.nodes[new_node.name] = new_node
        self.output_node = new_node
        new_node.set_artifacts_folder(self.artifacts_folder)

        # Inject logger
        new_node.logger = logging.getLogger(f"{self.logger.name}.<{new_node.name}>")

    def run(self, inp: Any = None) -> PipelineResult:
        """Accepts an input for the first node (must be a single value or None)

        Parameters
        ----------
        inp : Any, optional
            Input for the pipeline
            Must be provided if input node has one input
            Shouldn't be provided if input node has zero inputs

        Returns
        -------
        PipelineResult
            Data accumulated from the pipeline run
        """
        typs = [type(inp)]
        if not self.input_node.is_input_type_acceptable(typs):
            raise TypeError(f"Expected type {self.in_type} but got {typs}")

        history: Dict[str, str] = {"$input": inp}
        beginning_time = datetime.datetime.now()
        self.logger.info(f"Starting pipeline [at {beginning_time}]...")

        cache = {"$input": inp}

        try:
            return self.__run(inp, history, cache)
        except Exception as e:
            raise PipelineError("Pipeline failed with an exception", history) from e
        finally:
            self.__save_history(beginning_time, history)

    def resume(self, history_file_name: str, node_name: str) -> PipelineResult:
        """Resumes already finished execution from specified point
        Loads all saved data into the cache, purges cache that should be in the future relative to the specified node

        Parameters
        ----------
        history_file_name : str
            file that describes how data was saved

        node_name: str
            name of the node to resume from

        Returns
        -------
        PipelineResult
            Data accumulated from the pipeline run, includes old data from previous run as a part of the resumed run
        """
        history_dir = os.path.dirname(history_file_name)
        history_dir = os.path.abspath(history_dir)
        with open(history_file_name, "r") as f:
            history = json.loads(f.read())

        # TODO: Add pipeline structure check
        # TODO: Add reachability check
        # TODO: Optimize excessive caching

        if node_name not in history:
            raise ValueError(f"f{node_name} isn't present in history file {history_file_name} [{history}]")

        beginning_time = datetime.datetime.now()
        self.logger.info(f"Resuming pipeline [at {beginning_time}] [from {node_name}]...")

        cached_data: Dict[str, Any] = dict()

        # Load in cache all cachable outputs that are present in the history
        # and entry node itself
        for cached_node_name in self.__must_cache_output | {node_name}:
            if cached_node_name not in history or cached_node_name == "$input":
                continue

            dic_filename = history[cached_node_name]
            dic_filename = os.path.join(history_dir, dic_filename)
            cached_data[cached_node_name] = self.__load_data(cached_node_name, dic_filename)

        # Load original $input:
        if "$input" in self.__must_cache_output:
            cached_data["$input"] = self.in_type(history["$input"])

        # Set `resume` input
        inp = cached_data[node_name]

        old_execution_plan = self.execution_plan
        start_node_idx = self.execution_plan.index(self.nodes[node_name])
        self.execution_plan = self.execution_plan[start_node_idx + 1:]

        try:
            return self.__run(inp, history, cached_data)
        except Exception as e:
            raise PipelineError("Pipeline failed with an exception", history) from e
        finally:
            self.__save_history(beginning_time, history)
            self.execution_plan = old_execution_plan  # restore execution order

    def set_artifacts_folder(self, artifacts_folder: str):
        """Sets artifact folder for the pipeline and propagates it to all of its nodes

        Parameters
        ----------
        artifacts_folder : str
            path to the folder for the artifacts to be stored

        Notes
        -----
        This setting is a preference and may be ignored by the nodes

        """
        self.artifacts_folder = artifacts_folder
        for node in self.nodes.values():
            node.set_artifacts_folder(artifacts_folder)

    def __save_history(self, time: datetime.datetime, history: PipelineHistory):
        """Saves history file to the disk
        """
        if not self.store_intermediate_data:
            return

        pipeline_history_file = f"pipeline_{Pipeline.format_time(time)}.json"
        pipeline_history_file = os.path.join(self.artifacts_folder, pipeline_history_file)
        self.logger.info(f"Saving history [at {os.path.abspath(pipeline_history_file)}]")
        with open(pipeline_history_file, "w") as file:
            file.write(json.dumps(history))

    @staticmethod
    def get_timestamp_str() -> str:
        """Helper method to get current timestamp as a string suitable for use in file names"""
        return Pipeline.format_time(datetime.datetime.now())

    @staticmethod
    def format_time(time: datetime.datetime) -> str:
        """Helper method to get a timestamp as a string suitable for use in file names"""
        return time.strftime("%Y-%m-%d.%H-%M-%S")

    def __run(self, _input: Any, history: Dict[str, str], cache: Dict[str, Any]) -> PipelineResult:
        """
        Actually runs the pipeline

        Parameters
        ----------
        _input : Any
            input data for the first node

        history : Dict[str, str]
            A dictionary that contains mapping (node-name) -> (pipe-file-path)

        cache : Dict[str, Any]
            A dictionary that contains mapping (node-name) -> (produced-data)
            "$input" pseudo-node should be already in the cache

        Returns
        -------
        PipelineResult
        """
        self.check_prerequisites()

        if not os.path.exists(self.artifacts_folder):
            os.mkdir(self.artifacts_folder)

        statistic_dic: Dict[str, NodeStatistics] = {}

        prev_output: Any = _input
        prev_node_name: str = "$input"

        # Execution:
        for cur_node in self.execution_plan:
            # Construct input for the current node
            source_names = self.graph[cur_node.name]
            cur_input_data = []
            for s_name in source_names:
                cur_input_data.append(prev_output if s_name == prev_node_name else cache[s_name])

            # Type-check
            typs = [type(val) for val in cur_input_data]
            if not cur_node.is_input_type_acceptable(typs):
                raise TypeError(f"Input type(s) not acceptable by \"{cur_node.name}\" node\n"
                                f"expected types: {cur_node.in_types}, got: {typs}")
            try:
                self.logger.info(f"Running \"{cur_node.name}\" node")

                # Process data
                cur_node_start = time.time()
                del prev_output
                prev_output = cur_node.process(*cur_input_data)
                cur_node_processing_time = time.time() - cur_node_start

                prev_node_name = cur_node.name

                if cur_node.name in self.__must_cache_output:
                    cache[cur_node.name] = prev_output

                # Save data to disk before resuming
                cur_node_descriptor_start = time.time()
                if (self.store_intermediate_data and
                        (not cur_node.out_descriptor.is_optional() or self.store_optional_data)):
                    history[cur_node.name] = self.__store_data(cur_node, prev_output)
                cur_node_descriptor_time = time.time() - cur_node_descriptor_start

                # Collect statistics
                statistic_dic[cur_node.name] = NodeStatistics(
                    cur_node.name,
                    cur_node_processing_time,
                    cur_node_descriptor_time
                )

            except Exception as e:
                raise RuntimeError(f"Exception occurred during processing of node \"{cur_node.name}\"") from e

        return PipelineResult(prev_output, history, cache, statistic_dic)

    def __store_data(self, node: Node, data: Any) -> str:
        """Stores data of specific node to the disk
        Creates and fills a file

        Parameters
        ----------
        node : Node
            The node whose data is to be stored

        data: Any
            Data to be stored
            This data should be acceptable by the node's descriptor

        Returns
        -------
        str
            Path to the created file
        """
        dic = node.out_descriptor.store(data)
        dic_name = f"pipe_{node.name}_{self.get_timestamp_str()}.json"
        pipefile_path = os.path.abspath(os.path.join(self.artifacts_folder, dic_name))
        with open(pipefile_path, "w") as file:
            dic_str = json.dumps(dic)
            file.write(dic_str)

        return pipefile_path

    def __load_data(self, node_name: str, pipefile_path: str) -> Any:
        """Loads data from the disk for the specified node

        Parameters
        ----------
        node_name : str
            Name of the node to load data for
            node should be present in the pipeline

        pipefile_path : str
            Path to the file that holds the data
            The file should be produced by the same data descriptor

        Returns
        -------
        Any
            Restored data
        """
        node = self.nodes[node_name]
        with open(pipefile_path, "r") as file:
            dic_str = file.read()

        dic = json.loads(dic_str)
        return node.out_descriptor.load(dic)

