from __future__ import annotations

import datetime
import time
from collections import OrderedDict
from functools import lru_cache
from typing import Tuple, List

from scripts.coloring_pipeline import get_extended_coloring_pipeline
from server.render_colored_text import Coloring
from src.pipeline import Pipeline


# Pipeline configuration
def pipeline_preset(name: str, use_bidirectional_chaining: bool) -> Pipeline:
    pipeline = get_extended_coloring_pipeline(name)
    pipeline.assert_prerequisites()
    pipeline.force_caching("input-tokenized")
    pipeline.force_caching("$input")
    pipeline.store_optional_data = True
    pipeline.dont_timestamp_history = True
    all_chains: ChainingNode = pipeline.nodes["all-chains"]  # type: ignore
    all_chains.use_bidirectional_chaining = use_bidirectional_chaining
    return pipeline


# Create and configure Pipelines
unidir_pipeline = pipeline_preset("unidirectional", use_bidirectional_chaining=False)
bidir_pipeline = pipeline_preset("bidirectional", use_bidirectional_chaining=True)


def get_resume_points() -> List[str]:
    return list(unidir_pipeline.default_execution_order)


@lru_cache(5)
def color_text(text: str | None, store_data: bool, resume_node: str = "all-chains") -> Tuple[str, List[Coloring]]:
    if text is not None and store_data:
        unidir_pipeline.cleanup_file(unidir_pipeline.unstamped_history_filepath)
        bidir_pipeline.cleanup_file(bidir_pipeline.unstamped_history_filepath)

    unidir_pipeline.store_intermediate_data = store_data
    bidir_pipeline.store_intermediate_data = store_data

    coloring_variants = []

    start = time.time()

    # UNIDIRECTIONAL
    if text is None:
        result = unidir_pipeline.resume_from_disk(unidir_pipeline.unstamped_history_filepath, resume_node)
    else:
        result = unidir_pipeline.run(text)
    coloring_variants.append(
        Coloring(
            name="Unidirectional chaining",
            tokens=result.cache["input-tokenized"],
            pos2chain=result.last_node_result,
        )
    )
    stats = OrderedDict()
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
        Coloring(
            name="Bidirectional chaining",
            tokens=result.cache["input-tokenized"],
            pos2chain=result.last_node_result,
        )
    )
    stats["bidirectional"] = result.statistics

    # Preserve input
    if text is None:
        text = result.cache["$input"]
    del result

    seconds = time.time() - start

    print(f"Time taken to run: {datetime.timedelta(seconds=seconds)}")
    for i, (name, stats) in enumerate(stats.items()):
        print(f"Statistics (run {i+1}, {name})")
        for _, stat in stats.items():
            print(str(stat))

    return text, coloring_variants
