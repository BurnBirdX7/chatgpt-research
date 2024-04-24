from __future__ import annotations

import logging
import typing as t
from functools import lru_cache

from server.render_colored_text import Coloring, SourceColoring
from server.statistics_storage import storage
from src.pipeline import Pipeline
from src.pipeline.pipeline_group import PipelineGroup
from src.pipeline.pipeline_result import PipelineResult
from src.source_coloring_pipeline import SourceColoringPipeline


__all__ = ["get_resume_points", "run", "resume"]

logger = logging.getLogger(__name__)


# Pipeline configuration
def _pipeline_preset(name: str, use_bidirectional_chaining: bool) -> Pipeline:
    pipeline = SourceColoringPipeline.new_extended(name, use_bidirectional_chaining)
    pipeline.assert_prerequisites()

    # Force caching
    pipeline.force_caching("input-tokenized")
    pipeline.force_caching("$input")
    pipeline.force_caching("all-chains")
    if use_bidirectional_chaining:
        pipeline.force_caching("narrowed-sources-dict-tokenized")

    # Options
    pipeline.store_optional_data = True
    pipeline.dont_timestamp_history = True
    all_chains: ChainingNode = pipeline.nodes["all-chains"]  # type: ignore
    all_chains.use_bidirectional_chaining = use_bidirectional_chaining

    return pipeline


# GLOBALS
_unidir_pipeline = _pipeline_preset("unidirectional", use_bidirectional_chaining=False)
_bidir_pipeline = _pipeline_preset("bidirectional", use_bidirectional_chaining=True)
_pipeline_group = PipelineGroup("all-chains", [_unidir_pipeline, _bidir_pipeline])


def _collect_result(result: PipelineResult, is_first: bool) -> Coloring:
    if is_first:
        storage.sources = result.cache["narrowed-sources-dict-tokenized"]

    storage.chains[result.pipeline_name] = result.cache["all-chains"]

    if result.pipeline_name == "bidirectional":
        title = "Bidirectional chaining"
    else:
        title = "Unidirectional chaining"

    return SourceColoring(
        title=title,
        pipeline_name=result.pipeline_name,
        tokens=result.cache["input-tokenized"],
        chains=result.last_node_result,
    )


def get_resume_points() -> t.List[str]:
    return list(_unidir_pipeline.default_execution_order)


@lru_cache(5)
def run(input_text: str, override_data: bool) -> t.List[SourceColoring]:
    storage.clear_cache()
    logger.info(f"Coloring... override = {override_data}")

    _pipeline_group.override_data(override_data)

    colorings: t.Dict[str, SourceColoring]
    colorings, stats = _pipeline_group.run(input_text, _collect_result)

    print(stats.get_str())

    return list(colorings.values())


@lru_cache(5)
def resume(resume_point: str) -> t.Tuple[str, t.List[Coloring]]:
    storage.clear_cache()
    logger.info("Rerunning coloring... override = False")

    input_text: str = ""

    def __collect_text(result: PipelineResult, is_first: bool) -> Coloring:
        if is_first:
            nonlocal input_text
            input_text = result.cache["$input"]
        return _collect_result(result, is_first)

    colorings: t.Dict[str, SourceColoring]
    colorings, stats = _pipeline_group.resume(resume_point=resume_point, result_collector=__collect_text)

    print(stats.get_str())

    return input_text, list(colorings.values())
