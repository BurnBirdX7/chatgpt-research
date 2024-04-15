from __future__ import annotations

import datetime
import io
import logging
import time
from collections import OrderedDict, defaultdict
from functools import lru_cache
from typing import Tuple, List, Dict

import numpy as np
from matplotlib import pyplot as plt

from scripts.coloring_pipeline import get_extended_coloring_pipeline
from server.render_colored_text import Coloring
from src import ElasticChain
from src.pipeline import Pipeline


# Pipeline configuration
def pipeline_preset(name: str, use_bidirectional_chaining: bool) -> Pipeline:
    pipeline = get_extended_coloring_pipeline(name)
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

# Dict [key] -> List[Chain]
chain_dicts: Dict[str, List[ElasticChain]] = defaultdict(list)

sources_dict: Dict[str, List[str]] = defaultdict(list)

input_tokenized: List[str] = []
logger = logging.getLogger(__name__)


def get_resume_points() -> List[str]:
    return list(unidir_pipeline.default_execution_order)


def get_chains_for_target_pos(target_pos: int, key: str) -> List[ElasticChain]:
    chains = chain_dicts[key]
    return [chain for chain in chains if chain.target_begin_pos <= target_pos < chain.target_end_pos]


def get_chains_for_source_pos(source_name: str, source_pos: int):
    chains = chain_dicts["bidirectional"]
    return [
        chain
        for chain in chains
        if chain.source == source_name and chain.source_begin_pos <= source_pos < chain.source_end_pos
    ]


def plot_chains_likelihoods(target_pos: int, target_likelihood: float, chains: List[ElasticChain]) -> bytes:
    likelihoods = np.array([chain.get_target_likelihood(target_pos) for chain in chains])

    if (likelihoods < 0).any():
        raise RuntimeError("Found negative likelihood")

    img = io.BytesIO()

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(likelihoods, bins=16, range=(0, 1.0), color="grey")
    ax.axvline(x=np.max(likelihoods), color="r")
    ax.axvline(x=np.mean(likelihoods), color="g")
    ax.axvline(x=target_likelihood, color="b", linestyle="--")

    text_x = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
    text_y = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2

    ax.text(text_x, text_y, f"Chain count: {len(chains)}")

    ax.set_xlabel("likelihood")
    ax.set_ylabel("frequency")
    fig.savefig(img, format="png")
    plt.close(fig)
    img.seek(0)
    return img.read()


def _get_top10_target_chains(target_pos: int, key: str) -> List[ElasticChain]:
    chains = get_chains_for_target_pos(target_pos, key)
    chains = sorted(chains, reverse=True, key=lambda chain: chain.get_score())
    return list(chains[:10])


def _chain2dict(chain: ElasticChain) -> dict[str, str | int | float]:
    return {
        "text": "".join(input_tokenized[chain.target_begin_pos : chain.target_end_pos]),
        "score": chain.get_score(),
        "len": len(chain),
        "debug": str(chain),
    }


@lru_cache(200)
def get_top10_target_chains(target_pos: int, key: str) -> list[dict]:
    chains = _get_top10_target_chains(target_pos, key)

    return [_chain2dict(chain) for chain in chains]


@lru_cache(200)
def plot_pos_likelihoods(target_pos: int, target_likelihood: float, key: str) -> bytes:
    return plot_chains_likelihoods(target_pos, target_likelihood, get_chains_for_target_pos(target_pos, key))


def _get_top10_source_chains(key: str, source_name: str, source_pos: int) -> Tuple[str, list[ElasticChain]]:
    chains = get_chains_for_source_pos(source_name, source_pos)
    chains = sorted(chains, reverse=True, key=lambda chain: chain.get_score())
    tok = sources_dict[source_name][source_pos]
    return tok, list(chains[:10])


def get_top10_source_chains(key: str, source_name: str, source_pos: int) -> dict:
    tok, chains = _get_top10_source_chains(key, source_name, source_pos)

    return {"token": tok, "chains": [_chain2dict(chain) for chain in chains]}


@lru_cache(5)
def color_text(text: str | None, override_data: bool, resume_node: str = "all-chains") -> Tuple[str, List[Coloring]]:
    plot_pos_likelihoods.cache_clear()
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
        Coloring(
            title="Unidirectional chaining",
            pipeline_name=unidir_pipeline.name,
            tokens=result.cache["input-tokenized"],
            chains=result.last_node_result,
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
            chains=result.last_node_result,
        )
    )
    chain_dicts[bidir_pipeline.name] = result.cache["all-chains"]
    stats["bidirectional"] = result.statistics
    global sources_dict
    sources_dict = result.cache["narrowed-sources-dict-tokenized"]

    # Preserve input
    global input_tokenized
    input_tokenized = result.cache["input-tokenized"]
    if text is None:
        text = result.cache["$input"]
    del result

    seconds = time.time() - start

    print(f"Time taken to run: {datetime.timedelta(seconds=seconds)}")
    for i, (name, stats) in enumerate(stats.items()):
        print(f"Statistics (run {i + 1}, {name})")
        for _, stat in stats.items():
            print(str(stat))

    return text, coloring_variants
