import logging
from functools import lru_cache
import typing as t

import numpy as np
import numpy.typing as npt

from server.render_colored_text import Coloring, ScoreColoring
from server.statistics_storage import storage
from src.pipeline.pipeline_group import PipelineGroup
from src.pipeline.pipeline_result import PipelineResult
from src.score_coloring_pipeline import ScoreColoringPipeline


__all__ = ["run", "resume"]

logger = logging.getLogger(__name__)


# Pipeline configuration
def pipeline_preset(
    name: str, score_func: t.Callable[[t.List[float]], np.float32] | None = None
) -> ScoreColoringPipeline:
    pipeline = ScoreColoringPipeline(name)

    if score_func is not None:
        pipeline.nodes["scores"].score_func = score_func  # type: ignore

    # Force caching
    pipeline.force_caching("input-tokenized")
    pipeline.force_caching("$input")
    pipeline.force_caching("wide-chains")
    pipeline.force_caching("colors")

    # Options
    pipeline.store_optional_data = True
    pipeline.dont_timestamp_history = True

    return pipeline


def _reversed_mean(slice_: t.List[float]) -> np.float32:
    return 1 - np.exp(np.log(1.000001 - np.array(slice_)).sum() / len(slice_))


def _max(slice_: t.List[float]) -> np.float32:
    return np.max(slice_).astype(np.float32)


def _wrap(cut: int, func: t.Callable[[t.List[float]], np.float32]):
    def _func(slice_: t.List[float]) -> np.float32:
        cut_slice = np.sort(slice_).tolist()[:-cut]
        return func(cut_slice)

    return _func


# GLOBAL
_default_score_pipeline = pipeline_preset("geometric_mean")
_max_score_pipeline = pipeline_preset("max", _max)
_reversed_score_pipeline = pipeline_preset("reversed_geometric_mean", _reversed_mean)

first_pipelines = [_default_score_pipeline, _max_score_pipeline, _reversed_score_pipeline]

logging.basicConfig(level=logging.DEBUG)

for pipeline in list(first_pipelines):  # list(...) to make a copy of the list
    func = pipeline.nodes["scores"].score_func  # type: ignore

    for i in range(1, 6, 2):
        new_pipeline = pipeline_preset(pipeline.name + f"_(cut_top_{i})", _wrap(i, func))
        logger.debug(f"{pipeline.name} --> {new_pipeline.name}")
        first_pipelines.append(new_pipeline)

_pipeline_group = PipelineGroup("scores", first_pipelines)


def get_resume_points() -> t.List[str]:
    return list(_default_score_pipeline.default_execution_order)


def _collect_result(result: PipelineResult, _: bool) -> Coloring:
    storage.chains[result.pipeline_name] = result.cache["wide-chains"]

    title = result.pipeline_name[0].upper() + result.pipeline_name[1:].replace("_", " ") + " score coloring"

    return ScoreColoring(
        title=title,
        pipeline_name=result.pipeline_name,
        tokens=result.cache["input-tokenized"],
        scores=result.last_node_result,
        colors=result.cache["colors"],
    )


@lru_cache(1)
def run(text: str, override_data: bool) -> t.List[Coloring]:
    storage.clear_cache(map(lambda p: p.name, _pipeline_group.pipelines))
    logger.info(f"Coloring scores... override = {override_data}")

    _pipeline_group.override_data(override_data)

    colorings: t.Dict[str, Coloring]
    colorings, stats = _pipeline_group.run(text, _collect_result)

    print(stats.get_str())

    return list(colorings.values())


def resume(resume_point: str) -> t.Tuple[str, t.List[Coloring]]:
    storage.clear_cache(map(lambda p: p.name, _pipeline_group.pipelines))
    logger.info("Rerunning coloring... override = False")

    input_text: str = ""

    def __collect_text(result: PipelineResult, is_first: bool) -> Coloring:
        if is_first:
            nonlocal input_text
            input_text = result.cache["$input"]
        return _collect_result(result, is_first)

    colorings: t.Dict[str, ScoreColoring]
    colorings, stats = _pipeline_group.resume(resume_point=resume_point, result_collector=__collect_text)

    print(stats.get_str())

    return input_text, list(colorings.values())
