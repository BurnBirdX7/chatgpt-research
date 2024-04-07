import copy
import datetime
from dataclasses import dataclass
from typing import Dict, Any

PipelineHistory = Dict[str, str]
"""
Contains information about the data saved on to disk, after pipeline run

<node name> -> <pipe_file>

Pipe files are files filled by data descriptors, they store information required to restore a node's output
Pipe file can store output itself in case if simple data (str->str mappings or integers)
    or paths to files that store actual data
"""


@dataclass
class NodeStatistics:
    """
    Time statistics of each node that was called during the pipeline run
    """
    name: str
    node_seconds: float
    descriptor_seconds: float

    def __str__(self):
        return (
            f"NodeStatistics(\"{self.name}\", "
            f"time: {datetime.timedelta(seconds=self.node_seconds)}, "
            f"descriptor time: {datetime.timedelta(seconds=self.descriptor_seconds)})"
        )


@dataclass
class PipelineResult:
    """
    Collection of data produced by a pipeline
    """

    last_node_result: Any
    history: PipelineHistory
    cache: Dict[str, Any]
    statistics: Dict[str, NodeStatistics]

    def copy(self):
        return copy.deepcopy(self)
