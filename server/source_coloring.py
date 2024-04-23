from __future__ import annotations

import datetime
import logging
import time
import typing as t
from collections import OrderedDict
from functools import lru_cache

from server.render_colored_text import Coloring, SourceColoring
from server.statistics_storage import storage
from src.pipeline import Pipeline
from src.source_coloring_pipeline import SourceColoringPipeline


# Pipeline configuration
def pipeline_preset(name: str, use_bidirectional_chaining: bool) -> Pipeline:
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
unidir_pipeline = pipeline_preset("unidirectional", use_bidirectional_chaining=False)
bidir_pipeline = pipeline_preset("bidirectional", use_bidirectional_chaining=True)

logger = logging.getLogger(__name__)


def get_resume_points() -> t.List[str]:
    return list(unidir_pipeline.default_execution_order)


@lru_cache(5)
def source_color_text(
    text: str | None, override_data: bool, resume_node: str = "all-chains"
) -> t.Tuple[str, t.List[SourceColoring]]:
    storage.clear_cache()

    stats = OrderedDict()

    logger.info(f"Coloring.... override = {override_data}")

    if override_data:
        unidir_pipeline.cleanup_file(unidir_pipeline.unstamped_history_filepath)
        bidir_pipeline.cleanup_file(bidir_pipeline.unstamped_history_filepath)

    unidir_pipeline.store_intermediate_data = override_data
    bidir_pipeline.store_intermediate_data = override_data

    coloring_variants = []

    start = time.time()

    # UNIDIRECTIONAL
    if text is None:
        result = unidir_pipeline.resume_from_disk(unidir_pipeline.unstamped_history_filepath, resume_node)
    else:
        result = unidir_pipeline.run(text)
    coloring_variants.append(
        SourceColoring(
            title="Unidirectional chaining",
            pipeline_name=unidir_pipeline.name,
            tokens=result.cache["input-tokenized"],
            chains=result.last_node_result,
        )
    )
    storage.chains[unidir_pipeline.name] = result.cache["all-chains"]
    stats["unidirectional"] = result.statistics

    # BIDIRECTIONAL
    if text is None:
        exec_order = bidir_pipeline.default_execution_order
        if exec_order.index(resume_node) > exec_order.index("all-chains"):
            result = bidir_pipeline.resume_from_disk(bidir_pipeline.unstamped_history_filepath, resume_node)
        else:
            result = bidir_pipeline.resume_from_cache(result, resume_node)

    else:
        result = bidir_pipeline.resume_from_cache(result, "all-chains")

    coloring_variants.append(
        SourceColoring(
            title="Bidirectional chaining",
            pipeline_name=bidir_pipeline.name,
            tokens=result.cache["input-tokenized"],
            chains=result.last_node_result,
        )
    )
    storage.chains[bidir_pipeline.name] = result.cache["all-chains"]
    stats["bidirectional"] = result.statistics
    storage.sources = result.cache["narrowed-sources-dict-tokenized"]

    # Preserve input
    storage.input_tokenized = result.cache["input-tokenized"]
    if text is None:
        text = result.cache["$input"]
    del result

    seconds = time.time() - start

    print(f"Time taken to run: {datetime.timedelta(seconds=seconds)}")
    for i, (name, stats) in enumerate(stats.items()):
        print(f"Statistics (run {i + 1}, {name})")
        for _, stat in stats.items():
            print(str(stat))

    return t.cast(str, text), coloring_variants
