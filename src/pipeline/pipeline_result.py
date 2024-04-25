from __future__ import annotations

import copy
import datetime
import textwrap
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Any

__all__ = ["PipelineHistory", "PipelineResult", "NodeStatistics"]

PipelineHistory = Dict[str, str]
"""
Contains information about the data saved on to disk, after pipeline run

<node name> -> <pipe_file>

Pipe files are files filled by data descriptors, they store information required to restore a node's output
Pipe file can store output itself in case if simple data (str->str mappings or integers)
    or paths to files that store actual data
"""


@dataclass
class PipelineResult:
    """
    Collection of data produced by a pipeline
    """

    pipeline_name: str
    last_node_result: Any
    history: PipelineHistory
    cache: Dict[str, Any]
    statistics: PipelineStatistics

    def copy(self):
        return copy.deepcopy(self)


@dataclass
class PipelineStatistics:
    pipeline_name: str
    all_seconds: float
    prerequisite_seconds: float
    nodes: OrderedDict[str, NodeStatistics]

    @staticmethod
    def start(pipeline_name: str) -> PipelineStatisticsCollector:
        return PipelineStatisticsCollector(pipeline_name)

    @staticmethod
    def render_time(time_: float) -> str:
        return NodeStatistics.render_time(time_)

    def get_str(self):
        d_all = self.render_time(self.all_seconds)
        d_prereq = self.render_time(self.prerequisite_seconds)

        nodes_str = "\n".join([node.get_str() for node in self.nodes.values()])

        return (
            f"Pipeline Stats <{self.pipeline_name}>:\n"
            f"    all time           : {d_all}\n"
            f"    prerequisite check : {d_prereq}\n"
            f"{textwrap.indent(nodes_str, '    ')}"
        )


@dataclass
class NodeStatistics:
    """
    Time statistics of each node that was called during the pipeline run
    """

    name: str
    node_seconds: float
    descriptor_seconds: float
    all_seconds: float

    @staticmethod
    def start(name: str) -> NodeStatisticsCollector:
        """Convenience method for creating node statistics collector"""
        return NodeStatisticsCollector(name)

    @staticmethod
    def render_time(time_: float) -> str:
        minutes, seconds = divmod(time_, 60)
        if minutes > 0:
            return f"{int(minutes)} min, {seconds:.2f} sec"
        else:
            return f"{seconds:.2f} sec"

    def get_str(self) -> str:
        return (
            f"Node Stats <{self.name}>:\n"
            f"    all        : {self.render_time(self.all_seconds)}\n"
            f"    node       : {self.render_time(self.node_seconds)}\n"
            f"    descriptor : {self.render_time(self.descriptor_seconds)}"
        )


class NodeStatisticsCollector:
    def __init__(self, node_name: str):
        self.node_name = node_name

        self.start_time: float = time.time()
        self.end_time: float = 0

        self.produce_start_time: float = 0
        self.produce_end_time: float = 0
        self.descriptor_start_time: float = 0
        self.descriptor_end_time: float = 0

    def produce_start(self):
        self.produce_start_time = time.time()

    def produce_end(self):
        self.produce_end_time = time.time()

    def descriptor_start(self):
        self.descriptor_start_time = time.time()

    def descriptor_end(self):
        self.descriptor_end_time = time.time()

    def get(self) -> NodeStatistics:
        self.end_time = time.time()
        all_time = self.end_time - self.start_time
        node_sec = self.produce_end_time - self.produce_start_time
        desc_sec = self.descriptor_end_time - self.descriptor_start_time

        return NodeStatistics(
            name=self.node_name,
            node_seconds=node_sec,
            descriptor_seconds=desc_sec,
            all_seconds=all_time,
        )


class PipelineStatisticsCollector:
    def __init__(self, name: str) -> None:
        self.pipeline_name = name
        self._start_time: float = time.time()
        self._nodes_start_time: float = 0.0
        self._nodes: OrderedDict[str, NodeStatistics] = OrderedDict()

    def nodes_started(self):
        self._nodes_start_time = time.time()

    def add_node_stats(self, node_stat: NodeStatistics):
        self._nodes[node_stat.name] = node_stat

    def get(self) -> PipelineStatistics:
        end_time = time.time()

        return PipelineStatistics(
            pipeline_name=self.pipeline_name,
            all_seconds=end_time - self._start_time,
            prerequisite_seconds=self._nodes_start_time - self._start_time,
            nodes=self._nodes,
        )
