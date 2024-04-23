import logging
import time
from functools import lru_cache
from typing import Tuple, List

from server.render_colored_text import Coloring, ScoreColoring
from server.statistics_storage import storage
from src.score_coloring_pipeline import ScoreColoringPipeline


# Pipeline configuration
def pipeline_preset() -> ScoreColoringPipeline:
    pipeline = ScoreColoringPipeline("wide-score")
    pipeline.assert_prerequisites()

    # Force caching
    pipeline.force_caching("input-tokenized")
    pipeline.force_caching("$input")
    pipeline.force_caching("wide-chains")
    pipeline.force_caching("colors")

    # Options
    pipeline.store_optional_data = True
    pipeline.dont_timestamp_history = True

    return pipeline


# GLOBAL
score_pipeline = pipeline_preset()
logger = logging.getLogger(__name__)


@lru_cache(5)
def score_color_text(text: str | None, override_data: bool) -> Tuple[str, List[Coloring]]:
    storage.clear_cache()

    logger.info(f"Coloring scores... override = {override_data}")

    if override_data:
        score_pipeline.cleanup_file(score_pipeline.unstamped_history_filepath)
    score_pipeline.store_intermediate_data = override_data

    if text is None:
        raise ValueError("Resuming is not supported")

    result = score_pipeline.run(text)

    storage.chains[score_pipeline.name] = result.cache["wide-chains"]

    coloring = ScoreColoring(
        title="Score coloring",
        pipeline_name=score_pipeline.name,
        tokens=result.cache["input-tokenized"],
        scores=result.last_node_result,
        colors=result.cache["colors"],
    )

    return text, [coloring]
