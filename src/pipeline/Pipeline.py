from __future__ import annotations

import datetime
import json
import os.path
from typing import TypeVar, Any, cast

from src.pipeline.Block import Block

T = TypeVar('T')


class Pipeline:
    def __init__(self: "Pipeline"):
        self.pipeline = list[Block]()
        self.in_type: type | None = None
        self.artifacts_folder = "pipe-artifacts"
        self.name_bucket = list[str]()

    def add_block(self, block: Block) -> "Pipeline":
        if block.name in self.name_bucket:
            raise ValueError(f"Block with name '{block.name}' already exists in the pipeline")

        if self.in_type is None:
            self.in_type = block.in_type

        output_type: type = self.__get_output_type()
        if not issubclass(output_type, block.in_type):
            raise TypeError(f"Pipeline's output type does not match block's input type."
                            f"\tPipeline's output type is \"{output_type}\" and "
                            f"input type of the block \"{block.name}\" has type \"{block.in_type}")

        self.pipeline.append(block)
        if not os.path.exists(self.artifacts_folder):
            os.mkdir(self.artifacts_folder)

        block.set_artifacts_folder(self.artifacts_folder)
        return self

    def run(self, inp: Any = None) -> Any:
        if self.in_type is None:
            raise ValueError(f"Can't run empty pipeline")

        expected_typ = cast(type, self.in_type)
        if not isinstance(inp, expected_typ):
            raise TypeError(f"Expected type {expected_typ} but got {type(inp)}")

        history = dict[str, str]()
        beginning_time = datetime.datetime.now()
        print(f"Starting pipeline [at {beginning_time}]...")

        try:
            return self.__run(inp, history)
        except Exception as e:
            print("Pipeline failed with an exception:")
            print(e)
            raise e
        finally:
            pipeline_history_file = f"pipeline_{Pipeline.format_time(beginning_time)}.json"
            print(f"Saving history [at {os.path.abspath(pipeline_history_file)}]")
            with open(pipeline_history_file, "w") as file:
                file.write(json.dumps(history))

    @staticmethod
    def get_timestamp_str() -> str:
        return Pipeline.format_time(datetime.datetime.now())

    @staticmethod
    def format_time(time: datetime.datetime) -> str:
        return time.strftime("%Y-%m-%d.%H-%M-%S")

    def __get_output_type(self) -> type:
        if self.in_type is None:
            raise ValueError("Pipeline's input type in undefined")

        if len(self.pipeline) == 0:
            return cast(type, self.in_type)
        return self.pipeline[-1].out_type

    def __run(self, inp: Any, history: dict[str, str]) -> Any:
        """
        Run the pipeline
        :param history: dictionary where key is the name of the block and the value is name of the file with essential information
        """

        last_data = inp
        for block in self.pipeline:
            block_in_type = block.in_type
            if not isinstance(last_data, block_in_type):
                raise TypeError(f"Input descriptor type does not match with supplied data type:"
                                f"{block_in_type} vs {type(last_data)}")

            # Acquire data:
            last_data = block.process(last_data)

            # Save data to disk before resuming
            dic = block.out_descriptor.store(last_data)
            dic_name = f"pipe_{block.name}_{self.get_timestamp_str()}.json"
            filename = os.path.abspath(os.path.join(self.artifacts_folder, dic_name))
            with open(filename, "w") as file:
                dic_str = json.dumps(dic)
                file.write(dic_str)

            history[block.name] = dic_name

        return last_data
