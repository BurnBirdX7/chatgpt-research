from __future__ import annotations

import datetime
import io
import time
from collections import OrderedDict, defaultdict
from functools import lru_cache
from typing import Tuple, List, Dict

import numpy as np
from matplotlib import pyplot as plt

from scripts.coloring_pipeline import get_extended_coloring_pipeline
from server.render_colored_text import Coloring
from src import Chain
from src.pipeline import Pipeline


# Pipeline configuration
def pipeline_preset(name: str, use_bidirectional_chaining: bool) -> Pipeline:
    pipeline = get_extended_coloring_pipeline(name)
    pipeline.assert_prerequisites()

    # Force caching
    pipeline.force_caching("input-tokenized")
    pipeline.force_caching("$input")
    pipeline.force_caching("all-chains")

    # Options
    pipeline.store_optional_data = True
    pipeline.dont_timestamp_history = True
    all_chains: ChainingNode = pipeline.nodes["all-chains"]  # type: ignore
    all_chains.use_bidirectional_chaining = use_bidirectional_chaining
    return pipeline


# GLOBALS
unidir_pipeline = pipeline_preset("unidirectional", use_bidirectional_chaining=False)
bidir_pipeline = pipeline_preset("bidirectional", use_bidirectional_chaining=True)
chain_dicts: Dict[str, List[Chain]] = defaultdict(list)


def get_resume_points() -> List[str]:
    return list(unidir_pipeline.default_execution_order)


def get_chains_for_pos(target_pos: int, key: str) -> List[Chain]:
    chains = chain_dicts[key]
    return [chain for chain in chains if chain.target_begin_pos <= target_pos < chain.target_end_pos]

def plot_chains_likelihoods(target_pos: int, target_likelihood: float, chains: List[Chain]) -> bytes:
    likelihoods = np.array([chain.get_target_likelihood(target_pos) for chain in chains])

    if (likelihoods < 0).any():
        raise RuntimeError("Found negative likelihood")

    img = io.BytesIO()

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(likelihoods, bins=16, range=(0, 1.0), color="grey")
    ax.axvline(x=np.max(likelihoods), color='r')
    ax.axvline(x=np.mean(likelihoods), color='g')
    ax.axvline(x=target_likelihood, color='b', linestyle='--')

    ax.set_xlabel('likelihood')
    ax.set_ylabel('frequency')
    fig.savefig(img, format='png')
    plt.close(fig)
    img.seek(0)
    return img.read()


@lru_cache(200)
def plot_pos_likelihoods(target_pos: int, target_likelihood: float, key: str) -> bytes:
    return plot_chains_likelihoods(target_pos, target_likelihood, get_chains_for_pos(target_pos, key))


@lru_cache(5)
def color_text(text: str | None, store_data: bool, resume_node: str = "all-chains") -> Tuple[str, List[Coloring]]:
    plot_pos_likelihoods.cache_clear()
    stats = OrderedDict()

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
            title="Unidirectional chaining",
            pipeline_name=unidir_pipeline.name,
            tokens=result.cache["input-tokenized"],
            pos2chain=result.last_node_result,
        )
    )
    chain_dicts[unidir_pipeline.name] = result.cache["all-chains"]
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
            title="Bidirectional chaining",
            pipeline_name=bidir_pipeline.name,
            tokens=result.cache["input-tokenized"],
            pos2chain=result.last_node_result,
        )
    )
    chain_dicts[bidir_pipeline.name] = result.cache["all-chains"]
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
