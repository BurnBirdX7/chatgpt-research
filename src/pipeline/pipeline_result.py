from __future__ import annotations

import copy
import datetime
import time
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
            other_seconds=all_time - node_sec - desc_sec,
        )


@dataclass
class NodeStatistics:
    """
    Time statistics of each node that was called during the pipeline run
    """

    name: str
    node_seconds: float
    descriptor_seconds: float
    other_seconds: float

    def __str__(self) -> str:

        time_ = []
        if self.node_seconds > 0:
            time_.append(f"time: {datetime.timedelta(seconds=self.node_seconds)}")
        if self.descriptor_seconds > 0:
            time_.append(
                f"descriptor time: {datetime.timedelta(seconds=self.descriptor_seconds)}"
            )
        if self.other_seconds > 0:
            time_.append(
                f"other time: {datetime.timedelta(seconds=self.other_seconds)})"
            )

        s = ", ".join(time_)

        return f'NodeStatistics("{self.name}", {s})'

    @staticmethod
    def start(name: str) -> NodeStatisticsCollector:
        """Convenience method for creating node statistics collector"""
        return NodeStatisticsCollector(name)

    def __add__(self, other: NodeStatistics):
        if not isinstance(other, NodeStatistics):
            return NotImplemented

        if self.name != other.name:
            raise ValueError("Adding statistics with different names")

        return NodeStatistics(
            name=self.name,
            node_seconds=self.node_seconds + other.node_seconds,
            descriptor_seconds=self.descriptor_seconds + other.descriptor_seconds,
            other_seconds=self.other_seconds + other.other_seconds,
        )


@dataclass
class PipelineResult:
    """
    Collection of data produced by a pipeline
    """

    pipeline_name: str
    last_node_result: Any
    history: PipelineHistory
    cache: Dict[str, Any]
    statistics: Dict[str, NodeStatistics]

    def copy(self):
        return copy.deepcopy(self)
