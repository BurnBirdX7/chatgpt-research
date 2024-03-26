import datetime
from dataclasses import dataclass
from typing import Dict, Any

PipelineHistory = Dict[str, str]

@dataclass
class NodeStatistics:
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
    last_node_result: Any
    history: PipelineHistory
    cache: Dict[str, Any]
    statistics: Dict[str, NodeStatistics]
