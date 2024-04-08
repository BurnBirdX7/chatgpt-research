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

    #
    # INIT SECTION
    #

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
        self.graph: Dict[str, List[str]] = {inp.name: ["$input"]}  # Edges point towards data source
        self.__default_execution_order: List[str] = [inp.name]
        self.__must_cache_output: Set[str] = set()  # Set of nodes whose output should be cached

        # Misc
        self.name = name
        self.logger = logging.getLogger(f"pipeline.{self.name}")

    #
    # PUBLIC METHODS AND PROPERTIES
    #

    @property
    def default_execution_order(self) -> List[str]:
        return list(self.__default_execution_order)

    def attach_back(self, new_node: Node) -> "Pipeline":
        """Attaches node to the end of the pipeline (to the last attached node)

        Parameters
        ----------
        new_node : Node
            The node to attach

        Examples
        --------
        >>> pipeline.attach(SomeNode("last_node"), "other_node")
        ... pipeline.attach_back(OtherNode("new_node"))

        Creates following chain of nodes
        `... -> [last_node] -> [new_node]`
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
        >>> pipeline.attach(new_node, "node1", "node2")

        Makes `new_node` a pipeline's output.
        Also makes outputs of blocks `node1` and `node2` inputs for the new node.`

        Parameters
        ----------
        new_node : Node
            A node to attach

        *input_names : str
            The names of the inputs for the new nodes
            Number of names should be equal the number of inputs of the node
            Types of outputs should be compatible with inputs of the node

        Returns
        -------
        Pipeline
            reference to self, provided for convineince
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

    def replace_node(self, new_node: Node):
        """Replaces already existing node.

        Parameters
        ----------
        new_node : Node
            The new node. ``new_node.name`` must evaluate to name of already exiting node in the pipeline.
            The node must have the same input types as the node that is being replaced.
            Node must have the same output type.

        Raises
        ------
        ValueError
            if there's no node with the name ``new_node.name``

        TypeError
            if input or output types are incompatible with the old ones


        Examples
        --------
        You may need to replace the node to reuse some computations.
        In this example we make a computation, replace node somewhere in the middle of the pipeline,
        and restaring the computation with the new node

        >>> pipeline = Pipeline(InpNode("inp_node"))
        ... # Some nodes
        ... pipeline.attach(SomeNode("the_node", param=126))
        ... # More nodes
        ... res1 = pipeline.run("Hello")
        ... pipeline.replace_node(OtherNode("the_node"))
        ... res2 = pipeline.resume_from_cache(res1, "the_node")

        If you have parametrized node, you should just override the parameter, if possible

        >>> pipeline = Pipeline(InpNode("inp_node"))
        ... # Some nodes
        ... pipeline.attach(SomeNode("the_node", param=126))
        ... # More nodes
        ... res1 = pipeline.run("Hello")
        ... pipeline.nodes["the_node"].param=621
        ... res2 = pipeline.resume_from_cache(res1, "the_node")

        Notes
        -----
        Method makes basic checks for input/output types before replacing node in the pipeline.
        Replacement process itself is very simple, as it just requires to replace value in ``self.nodes``
        """

        node_name = new_node.name
        if node_name not in self.nodes:
            raise ValueError(f"Pipeline has no node named {node_name}")

        old_node = self.nodes[node_name]

        if len(old_node.in_types) != len(new_node.in_types):
            raise TypeError("Different input length")

        for new_typ, old_typ in zip(new_node.in_types, old_node.in_types):
            if not issubclass(old_typ, new_typ):
                raise TypeError(f"Incompatible input types of old and new nodes: {old_node.in_types} vs {new_node.in_types}")

        if not issubclass(new_node.out_type, old_node.out_type):
            raise TypeError(f"Incompatible output types of old and new nodes: {old_node.out_type} vs {new_node.out_type}")

        self.nodes[node_name] = new_node

    def force_caching(self, node_name: str) -> None:
        """
        Force caching of node's output during pipeline execution
        After execution in may be retrieved from the PipelineResult object

        Parameters
        ----------
        node_name : str
            Name of the node whose output should be cached
        """
        if node_name in self.nodes:
            self.__must_cache_output.add(node_name)

    def prerequisites_check(self) -> Dict[str, str] | None:
        """
        Checks if all prerequisites are met

        Returns
        -------
        Dict[str, str] | None
            Dict is returned if some requirements aren't met. The dict contains [node_name] -> [description] pairs
            None is returned if all requirements are met
        """

        errs = {}
        for node in self.nodes.values():
            r = node.prerequisite_check()
            if r is not None:
                errs[node.name] = r

        if len(errs) == 0:
            return None
        return errs

    def assert_prerequisites(self):
        """Raises ValueError if prerequisites are not met
        """
        r = self.prerequisites_check()
        if r is not None:
            raise ValueError(f"Prerequisite check failed for some nodes: {r!s}")

    def run(self, inp: Any = None) -> PipelineResult:
        """Accepts an input for the first node (must be a single value or None)

        Parameters
        ----------
        inp : Any, optional
            Input for the pipeline.
            Must be provided if input node has one input.
            Shouldn't be provided if input node has zero inputs.

        Returns
        -------
        PipelineResult
            Data accumulated from the pipeline run
        """

        history: Dict[str, str] = {"$input": inp}
        beginning_time = datetime.datetime.now()
        self.logger.info(f"Starting pipeline [at {beginning_time}]...")

        cache = {"$input": inp}

        try:
            return self.__run(inp, history, cache, self.default_execution_order)
        except Exception as e:
            raise PipelineError("Pipeline failed with an exception", history) from e
        finally:
            self.__save_history(beginning_time, history)

    def resume_from_cache(self, pipeline_result: PipelineResult, node_name: str) -> PipelineResult:
        """Resumes already finished execution from specified point
        Reuses data from cache

        Parameters
        ----------
        pipeline_result : PipelineResult
            Result of previous execution

        node_name : str
            Name of the node to be executed first

        Returns
        -------
        PipelineResult
            Data accumulated from the pipeline run, includes old data from previous run as a part of the resumed run
        """

        node_index = self.__default_execution_order.index(node_name)
        execution_order = self.__default_execution_order[node_index:]
        previous_execution_order = self.__default_execution_order[:node_index]

        # Copy history before target node
        history: Dict[str, str] = {
            k: pipeline_result.history[k]
            for k in previous_execution_order
        }

        # Copy cache of outputs produces before target node
        cache: Dict[str, Any] = {
            k: pipeline_result.cache[k]
            for k in previous_execution_order
        }

        start_time = datetime.datetime.now()
        self.logger.info(f"Resuming pipeline from cache [time: {start_time}]...")

        try:
            return self.__run(None, history, cache, execution_order)
        except Exception as e:
            raise PipelineError("Pipeline failed with an exception", history) from e
        finally:
            self.__save_history(start_time, history)

    def resume_from_disk(self, historyfile_path: str, node_name: str) -> PipelineResult:
        """Resumes already finished execution from specified point
        Loads all saved data into the cache, purges cache that should be in the future relative to the specified node

        Parameters
        ----------
        historyfile_path : str
            Path to the file with history

        node_name: str
            Name of the node to be executed first

        Returns
        -------
        PipelineResult
            Data accumulated from the pipeline run, includes old data from previous run as a part of the resumed run
        """

        history_dir = os.path.dirname(historyfile_path)
        history_dir = os.path.abspath(history_dir)
        with open(historyfile_path, "r") as f:
            history = json.loads(f.read())

        start_time = datetime.datetime.now()
        self.logger.info(f"Resuming pipeline from disk [time: {start_time}] [file: {historyfile_path}] [node: {node_name}]...")

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
        start_node_idx = self.__default_execution_order.index(node_name)
        execution_order = self.__default_execution_order[start_node_idx:]

        try:
            return self.__run(None, history, cached_data, execution_order)
        except Exception as e:
            raise PipelineError("Pipeline failed with an exception", history) from e
        finally:
            self.__save_history(start_time, history)

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

    @staticmethod
    def get_timestamp_str() -> str:
        """Helper method to get current timestamp as a string suitable for use in file names"""
        return Pipeline.format_time(datetime.datetime.now())

    @staticmethod
    def format_time(time: datetime.datetime) -> str:
        """Helper method to get a timestamp as a string suitable for use in file names"""
        return time.strftime("%Y-%m-%d.%H-%M-%S")

    #
    # PRIVATE METHODS
    #

    def __set_node(self, new_node: Node, input_names: list[str]):
        """Sets up a new node, must be called for every new node

        Parameters
        ----------
        new_node : Node

        input_names : list[str]
            List of nodes that provide inputs for the new node

        Notes
        -----
        This method propagates artifacts folder to the node and injects new logger into it,
            so logs made previously may be different from ones that will be made after that
        """

        # Update graph and execution order
        self.__default_execution_order.append(new_node.name)
        self.graph[new_node.name] = input_names

        # General structure updated
        self.nodes[new_node.name] = new_node
        self.output_node = new_node
        new_node.set_artifacts_folder(self.artifacts_folder)

        # Inject logger
        new_node.logger = logging.getLogger(f"{self.logger.name}.<{new_node.name}>")

    def __save_history(self, time: datetime.datetime, history: PipelineHistory):
        """Saves history file to the disk
        Does nothing if ``self.store_intermediate_data`` is set to `False`

        Parameters
        ----------
        time : datetime.datetime
            Time that used to timestamp the history files

        history: PipelineHistory
            History to be saved

        """
        if not self.store_intermediate_data:
            return

        pipeline_history_file = f"pipeline_{self.name}_{Pipeline.format_time(time)}.json"
        pipeline_history_file = os.path.join(self.artifacts_folder, pipeline_history_file)
        self.logger.info(f"Saving history [path: {os.path.abspath(pipeline_history_file)}]")
        with open(pipeline_history_file, "w") as file:
            file.write(json.dumps(history))

    def __run(self,
              input_: Any,
              history: Dict[str, str],
              cache: Dict[str, Any],
              execution_order: List[str]) -> PipelineResult:
        """
        Actually runs the pipeline

        Parameters
        ----------
        input_ : Any
            input data for the first node

        history : Dict[str, str]
            A dictionary that contains mapping [node_name] -> [pipefile_path]

        cache : Dict[str, Any]
            A dictionary that contains mapping [node_name] -> [produced_data].
            "$input" pseudo-node should be already in the cache

        execution_order : List[str]
            Names of nodes in order of execution.

        Returns
        -------
        PipelineResult
        """
        self.assert_prerequisites()

        if not os.path.exists(self.artifacts_folder):
            os.mkdir(self.artifacts_folder)

        statistic_dic: Dict[str, NodeStatistics] = {}

        prev_output: Any = input_
        prev_node_name: str = "$input" if input_ is not None else None

        # Execution:
        for cur_node_name in execution_order:
            # Construct input for the current node
            cur_node = self.nodes[cur_node_name]
            source_names = self.graph[cur_node_name]
            cur_input_data = []
            for s_name in source_names:
                cur_input_data.append(prev_output if s_name == prev_node_name else cache[s_name])

            # Type-check
            typs = [type(val) for val in cur_input_data]
            if not cur_node.is_input_type_acceptable(typs):
                raise TypeError(f"Input type(s) not acceptable by \"{cur_node_name}\" node\n"
                                f"expected types: {cur_node.in_types}, got: {typs}")
            try:
                self.logger.info(f"Running \"{cur_node_name}\" node")

                # Process data
                stat_track = NodeStatistics.start(cur_node_name)
                stat_track.produce_start()
                prev_output = cur_node.process(*cur_input_data)
                stat_track.produce_end()

                prev_node_name = cur_node_name

                if cur_node_name in self.__must_cache_output:
                    cache[cur_node_name] = prev_output

                # Save data to disk before resuming
                stat_track.descriptor_start()
                self.__store_data(cur_node, prev_output, history)
                stat_track.descriptor_end()

                # Collect statistics
                statistic_dic[cur_node_name] = stat_track.get()

            except Exception as e:
                raise RuntimeError(f"Exception occurred during processing of node \"{cur_node.name}\"") from e

        return PipelineResult(self.name, prev_output, history, cache, statistic_dic)

    def __store_data(self, node: Node, data: Any, history: PipelineHistory) -> None:
        """Stores data of specific node to the disk. Creates and fills a file

        If ``self.store_intermediate_data`` is `False` data won't be saved.

        Data won't be saved if it's optional and ``self.store_optional_data`` is `False`
36
        Parameters
        ----------
        node : Node
            The node whose data is to be stored

        data : Any
            Data to be stored.
            This data should be acceptable by the node's descriptor

        history : PipelineHistory
            Dictionary to where path to produced pipefile will be saved
        """

        if (not self.store_intermediate_data and
                (node.out_descriptor.is_optional() or not self.store_optional_data)):
            return

        dic = node.out_descriptor.store(data)
        dic_name = f"pipe_{node.name}_{self.get_timestamp_str()}.json"
        pipefile_path = os.path.abspath(os.path.join(self.artifacts_folder, dic_name))
        with open(pipefile_path, "w") as file:
            dic_str = json.dumps(dic)
            file.write(dic_str)

        history[node.name] = pipefile_path

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

