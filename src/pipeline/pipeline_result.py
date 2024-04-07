from __future__ import annotations

import copy
import datetime
import time
from dataclasses import dataclass
from typing import Dict, Any

__all__ = [
    'PipelineHistory',
    'PipelineResult',
    'NodeStatistics'
]

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

        self.start_time = time.time()
        self.end_time = 0

        self.produce_start_time = 0
        self.produce_end_time = 0
        self.descriptor_start_time = 0
        self.descriptor_end_time = 0

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
            other_seconds=all_time - node_sec - desc_sec
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
        return (
            f"NodeStatistics(\"{self.name}\", "
            f"time: {datetime.timedelta(seconds=self.node_seconds)}, "
            f"descriptor time: {datetime.timedelta(seconds=self.descriptor_seconds)}, "
            f"other time: {datetime.timedelta(seconds=self.other_seconds)})"
        )

    @staticmethod
    def start(name: str) -> NodeStatisticsCollector:
        """Convenience method for creating node statistics collector
        """
        return NodeStatisticsCollector(name)


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
