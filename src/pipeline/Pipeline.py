from __future__ import annotations

import datetime
import json
import os.path
import types
from inspect import signature
from typing import Any, cast, Callable, Dict, Set, List

from src.pipeline.Block import Block

PipelineHistory = Dict[str, str]


class PipelineError(RuntimeError):
    def __init__(self, message: str, history: dict[str, str]) -> None:
        super().__init__(message)
        self._history = history


class Pipeline:
    """
    Class that helps streamline data processing pipelines
    """

    def __init__(self: "Pipeline", inp: Block):
        self.input_block = inp
        self.in_type = inp.in_type
        self.artifacts_folder = "pipe-artifacts"
        self.last_block = inp
        self.blocks: Dict[str, Block] = {inp.name: inp}  # All Blocks in the pipeline
        self.execution_order = [inp]

        self.__cache_output: Set[str] = set()  # Set of blocks whose output should be cached
        self.__graph: Dict[str, List[str]] = {inp.name: []}  # Data-flow graph
        self.__source_graph: Dict[str, List[str]] = {inp.name: ["$input"]}  # Edges point towards data source
        self.__merge_funcs: Dict[str, Callable] = dict()  # Functions that fold multiple inputs into one

    def attach(self, block: Block, input_name: str | None = None) -> "Pipeline":
        """
        Attaches block to the end of the pipeline
        ... -> [p] -> [block]

        Or to the specified block `i`
        ... -> [i] -> [p] -> ...
                |
                v
              [block]
        """

        # Important checks
        if block.name in self.__graph:
            raise ValueError(f"Block with name '{block.name}' already exists in the pipeline")

        input_type: type
        input_block: Block
        if input_name is None:
            input_block = self.last_block
        else:
            input_block = self.blocks[input_name]

        input_type = input_block.out_type
        if input_block is not self.last_block:
            self.__cache_output |= {input_block.name}

        if not block.is_type_acceptable(input_type):
            raise TypeError(f"Pipeline's output type is not acceptable by the block"
                            f"\tPipeline's output type is \"{input_type}\" and "
                            f"input type of the block \"{block.name}\" has type \"{block.in_type}")

        # Set block into place
        self.__set_block(block, [input_block])
        return self

    def merge(self,
              block: Block,
              input_names: list[str],
              merge_func: Callable,
              suppress_error: bool = False,
              ) -> "Pipeline":
        r"""
        Merges outputs of specified blocks and links it to 'block'

        ... -> [i1] --\
                       v
                    [block]
                       ^
        ... -> [i2] --/
        """

        # Important checks
        if block.name in self.__graph:
            raise ValueError(f"Block with name '{block.name}' already exists in the pipeline")

        inputs = list[Block]()
        for name in input_names:
            if name not in self.blocks:
                raise ValueError(f"Block with name '{block.name}' does not exist in the pipeline")

            inp_block = self.blocks[name]
            inputs.append(inp_block)

        # It's impossible to check for sure, so we check annotations to confirm correct typing
        merge_sig = signature(merge_func)
        if len(merge_sig.parameters) != len(inputs):
            raise ValueError(
                f"Merge function accepts {len(merge_sig.parameters)} args but {len(inputs)} inputs were provided")

        if isinstance(merge_func, types.LambdaType) and merge_func.__name__ == "<lambda>":
            print("[WARNING] Merge function is a lambda function, typechecking is impossible")
        elif not suppress_error:
            error_msg = ""
            for inp, merge_param in zip(inputs, merge_sig.parameters.values()):
                if not issubclass(inp.out_type, merge_param.annotation):
                    error_msg += f"\tParameter \"{merge_param.name}\" must be of type \"{inp.out_type}"
                    error_msg += f" but got type \"{merge_param.annotation}\n"

            if not block.is_type_acceptable(merge_sig.return_annotation):
                error_msg += (f"\tReturn type {merge_sig.return_annotation} "
                              f"is not acceptable by the block (expected {block.in_type})")

            if len(error_msg) > 0:
                error_msg = ("Merge function's annotated types are not acceptable by the block "
                             "or annotations are missing:\n") + error_msg
                raise TypeError(error_msg)

        # Set block into place
        self.__merge_funcs[block.name] = merge_func
        self.__set_block(block, inputs)
        self.__cache_output |= set(input_names)

        return self

    def __set_block(self, block: Block, inputs: list[Block]):
        def get_name(b: Block) -> str:
            return b.name

        names = list(map(get_name, inputs))

        self.execution_order.append(block)
        self.__source_graph[block.name] = names
        self.__graph[block.name] = []
        for name in names:
            self.__graph[block.name].append(name)

        self.blocks[block.name] = block
        self.last_block = block
        block.set_artifacts_folder(self.artifacts_folder)

    def run(self, inp: Any) -> tuple[Any, PipelineHistory]:
        if not isinstance(inp, self.in_type):
            raise TypeError(f"Expected type {self.in_type} but got {type(inp)}")

        history = dict[str, str]()
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
        for cached_block_name in self.__cache_output | {block_name}:
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
        idx = self.execution_order.index(self.blocks[block_name])
        self.execution_order = self.execution_order[idx+1:]

        try:
            return self.__run(inp, history, cached_data), history
        except Exception as e:
            raise PipelineError("Pipeline failed with an exception", history) from e
        finally:
            self.__save_history(beginning_time, history)
            self.execution_order = exec_order  # restore execution order

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

    def __run(self, inp: Any, history: Dict[str, str], cached_data: Dict[str, Any]) -> Any:
        """
        Run the pipeline
        :param history: dictionary where key is the name of the block and the value is name of the file with essential information
        """

        if not os.path.exists(self.artifacts_folder):
            os.mkdir(self.artifacts_folder)

        last_data: Any = inp
        input_data: Any = inp
        last_block: str = "$input"

        for block in self.execution_order:
            sources = self.__source_graph[block.name]
            if sources == [last_block]:  # Source list matches last block, use last data
                input_data = last_data
            else:  # If it doesn't match - load from cache
                cached_value = list()
                for source in sources:
                    cached_value.append(cached_data[source])
                if len(cached_value) > 1:
                    input_data = self.__merge_funcs[block.name](*cached_value)
                else:
                    input_data = cached_value[0]

            if not block.is_value_acceptable(input_data):
                raise TypeError(f"Input type not acceptable for {block.name}\n"
                                f"expected: {block.in_type}, got: {type(input_data)}")

            last_data = block.process(input_data)
            last_block = block.name

            if block.name in self.__cache_output:
                cached_data[block.name] = last_data

            # Save data to disk before resuming
            history[block.name] = self.__store_data(block, last_data)

        return last_data

    def __store_data(self, block: Block, data: Any) -> str:
        dic = block.out_descriptor.store(data)
        dic_name = f"pipe_{block.name}_{self.get_timestamp_str()}.json"
        filename = os.path.abspath(os.path.join(self.artifacts_folder, dic_name))
        with open(filename, "w") as file:
            dic_str = json.dumps(dic)
            file.write(dic_str)

        return dic_name

    def __load_data(self, block_name: str, dic_filename: str) -> Any:
        block = self.blocks[block_name]
        with open(dic_filename, "r") as file:
            dic_str = file.read()

        dic = json.loads(dic_str)
        return block.out_descriptor.load(dic)




