from __future__ import annotations

import datetime
import json
from typing import TypeVar, Any

from src.pipeline.BaseDataDescriptor import BaseDataDescriptor
from src.pipeline.Block import Block

T = TypeVar('T')


class Pipeline:
    def __init__(self):
        self.pipeline = list[Block]()
        self.starting_descriptor: type[BaseDataDescriptor] | None = None

    def add_block(self, block: Block) -> "Pipeline":
        if self.starting_descriptor is None:
            self.starting_descriptor = block.in_descriptor_type

        last_desc_typ = self.__get_last_data_descriptor_type()
        if block.in_descriptor_type != last_desc_typ:
            raise TypeError(f"Descriptor types do not match."
                            f"\tLast descriptor type in chain is \"{last_desc_typ}\" and "
                            f"descriptor type of supplied block is \"{block.in_descriptor_type}")

        self.pipeline.append(block)
        return self

    def run(self, inp: Any = None) -> Any:
        expected_typ = self.starting_descriptor.get_data_type()
        if type(inp) == expected_typ:
            raise TypeError(f"Expected type {expected_typ} but got {type(inp)}")

        history = dict[str, str]()
        beginning_time = datetime.datetime.now()
        print(f"Starting pipeline [at {beginning_time}]...")

        try:
            return self.__run(inp, history)
        except Exception as e:
            print("Pipeline failed with an exception:")
            print(e)
        finally:
            pipeline_history_file = f"pipeline_{Pipeline.format_time(beginning_time)}.json"
            print(f"Saving history [at {pipeline_history_file}]")
            with open(pipeline_history_file, "w") as file:
                file.write(json.dumps(history))

    @staticmethod
    def get_timestamp_str() -> str:
        return Pipeline.format_time(datetime.datetime.now())

    @staticmethod
    def format_time(time: datetime.datetime) -> str:
        return time.strftime("%Y-%m-%d.%H-%M-%S")

    def __get_last_data_descriptor_type(self) -> type[BaseDataDescriptor] | None:
        if len(self.pipeline) == 0:
            return self.starting_descriptor
        return self.pipeline[-1].out_descriptor_type

    def __run(self, inp: Any, history: dict[str, str]) -> Any:
        """
        Run the pipeline
        :param history: dictionary where key is the name of the block and the value is name of the file with essential information
        """

        last_data = inp
        for block in self.pipeline:
            inp_desc = block.in_descriptor_type
            inp_desc_data_type = inp_desc.get_data_type()
            if inp_desc_data_type != type(last_data):
                raise TypeError(f"Input descriptor type does not match with supplied data type:"
                                f"{inp_desc_data_type} vs {type(last_data)}")

            # Acquire data:
            last_data = block.process(last_data)

            # Save data to disk before resuming
            dic = block.out_descriptor.store(last_data)
            dic_name = f"pipe_{block.block_name}_{self.get_timestamp_str()}.json"
            with open(dic_name, "w") as file:
                dic_str = json.dumps(dic)
                file.write(dic_str)

            history[block.block_name] = dic_name

        return last_data
