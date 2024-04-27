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
    name: str, score_func: t.Callable[[npt.NDArray[np.float32]], np.float32] | None = None
) -> ScoreColoringPipeline:
    pipeline = ScoreColoringPipeline(name)
    pipeline.assert_prerequisites()

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


def _reversed_mean(slice_: npt.NDArray[np.float32]) -> np.float32:
    return 1 - np.exp(np.log(1 - slice_).sum() / len(slice_))


# GLOBAL
_default_score_pipeline = pipeline_preset("geometric_mean")
_max_score_pipeline = pipeline_preset("max", np.max)
_reversed_score_pipeline = pipeline_preset("reversed_geometric_mean", _reversed_mean)

_pipeline_group = PipelineGroup("wide-chains", [_default_score_pipeline, _max_score_pipeline, _reversed_score_pipeline])


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


@lru_cache(5)
def run(text: str, override_data: bool) -> t.List[Coloring]:
    storage.clear_cache()
    logger.info(f"Coloring scores... override = {override_data}")

    _pipeline_group.override_data(override_data)

    colorings: t.Dict[str, Coloring]
    colorings, stats = _pipeline_group.run(text, _collect_result)

    print(stats.get_str())

    return list(colorings.values())


def resume(resume_point: str) -> t.Tuple[str, t.List[Coloring]]:
    storage.clear_cache()
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
