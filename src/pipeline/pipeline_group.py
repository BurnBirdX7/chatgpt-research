import dataclasses
import datetime
import textwrap
import time
import typing as t
from collections import OrderedDict

from .pipeline import Pipeline
from .pipeline_result import PipelineResult, NodeStatistics, PipelineStatistics


PipelineResultCollector = t.Callable[[PipelineResult, bool], t.Any]


def _noop(result: PipelineResult, is_first: bool) -> t.Any:
    return result


class PipelineGroup:

    def __init__(self, diverge_point: str, pipelines: t.Sequence[Pipeline]):
        self.diverge_point = diverge_point
        self.pipelines = pipelines

        for pipeline in self.pipelines:
            pipeline.dont_timestamp_history = True

    def for_each(self, func: t.Callable[[Pipeline], None]):
        for pipeline in self.pipelines:
            func(pipeline)

    def override_data(self, val: bool = True):
        for pipeline in self.pipelines:
            if val:
                pipeline.cleanup_file(pipeline.unstamped_history_filepath)
            pipeline.store_intermediate_data = val

    def run(
        self, inp: t.Any = None, result_collector: PipelineResultCollector = _noop
    ) -> t.Tuple[t.Dict[str, t.Any], "GroupStatistics"]:
        first_pipeline = self.pipelines[0]
        other_pipelines = self.pipelines[1:]

        results = OrderedDict()
        statistics = GroupStatistics()

        start = time.time()

        result = first_pipeline.run(inp)
        results[first_pipeline.name] = result_collector(result, True)
        statistics.pipelines.append(result.statistics)

        for pipeline in other_pipelines:
            result = pipeline.resume_from_cache(result, self.diverge_point)
            results[pipeline.name] = result_collector(result, False)
            statistics.pipelines.append(result.statistics)

        statistics.all_seconds = time.time() - start

        return results, statistics

    def resume(
        self, resume_point: str | None, result_collector: PipelineResultCollector = _noop
    ) -> t.Tuple[t.Dict[str, t.Any], "GroupStatistics"]:
        if resume_point is None:
            resume_point = self.diverge_point

        first_pipeline = self.pipelines[0]
        other_pipelines = self.pipelines[1:]

        exec_order = first_pipeline.default_execution_order
        load_from_disk = exec_order.index(resume_point) > exec_order.index(self.diverge_point)

        results = OrderedDict()
        statistics = GroupStatistics()

        start = time.time()

        result = first_pipeline.resume_from_disk(first_pipeline.unstamped_history_filepath, resume_point)
        results[first_pipeline.name] = result_collector(result, True)
        statistics.pipelines.append(result.statistics)

        for pipeline in other_pipelines:
            if load_from_disk:
                result = pipeline.resume_from_disk(pipeline.unstamped_history_filepath, self.diverge_point)
            else:
                result = pipeline.resume_from_cache(result, resume_point)
            results[pipeline.name] = result_collector(result, False)
            statistics.pipelines.append(result.statistics)

        statistics.all_seconds = time.time() - start

        return results, statistics


@dataclasses.dataclass
class GroupStatistics:
    all_seconds: float = 0.0
    pipelines: t.List[PipelineStatistics] = dataclasses.field(default_factory=list)

    def get_str(self) -> str:
        d_all = PipelineStatistics.render_time(self.all_seconds)

        return f"Group Stats:\n\tall time : {d_all}\n" + textwrap.indent(
            "\n".join(stat.get_str() for stat in self.pipelines), "    "
        )
