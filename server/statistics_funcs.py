from __future__ import annotations

import io
from functools import lru_cache
from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt

from server.statistics_storage import storage
from src.chaining import Chain


def get_chains_for_target_pos(target_pos: int, key: str) -> List[Chain]:
    chains = storage.chains[key]
    return [chain for chain in chains if chain.target_begin_pos <= target_pos < chain.target_end_pos]


def get_chains_for_source_pos(key: str, source_name: str, source_pos: int):
    chains = storage.chains[key]
    return [
        chain
        for chain in chains
        if chain.source == source_name and chain.source_begin_pos <= source_pos < chain.source_end_pos
    ]


def plot_chains_likelihoods(target_pos: int, target_likelihood: float, chains: List[Chain]) -> bytes:
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
    ax.set_yscale("log")

    fig.savefig(img, format="png")
    plt.close(fig)
    img.seek(0)
    return img.read()


def _get_top10_target_chains(target_pos: int, key: str) -> List[Chain]:
    chains = get_chains_for_target_pos(target_pos, key)
    chains = sorted(chains, reverse=True, key=lambda chain: chain.get_score())
    return list(chains[:10])


def _chain2dict(chain: Chain) -> dict[str, str | int | float]:
    return {
        "text": "".join(storage.input_tokenized[chain.target_begin_pos : chain.target_end_pos]),
        "score": chain.get_score(),
        "len": chain.target_len(),
        "debug": str(chain),
    }


def _get_top10_source_chains(key: str, source_name: str, source_pos: int) -> Tuple[str, list[Chain]]:
    chains = get_chains_for_source_pos(key, source_name, source_pos)
    chains = sorted(chains, reverse=True, key=lambda chain: chain.get_score())
    tok = storage.sources[source_name][source_pos]
    return tok, list(chains[:10])


@lru_cache(200)
def get_top10_target_chains(target_pos: int, key: str) -> list[dict]:
    chains = _get_top10_target_chains(target_pos, key)

    return [_chain2dict(chain) for chain in chains]


@lru_cache(200)
def plot_pos_likelihoods(target_pos: int, target_likelihood: float, key: str) -> bytes:
    return plot_chains_likelihoods(target_pos, target_likelihood, get_chains_for_target_pos(target_pos, key))


@lru_cache(200)
def get_top10_source_chains(key: str, source_name: str, source_pos: int) -> dict:
    tok, chains = _get_top10_source_chains(key, source_name, source_pos)

    return {"token": tok, "chains": [_chain2dict(chain) for chain in chains]}


# Register functions that store cache, to allow clearing that cache on request
storage.register_func(get_top10_target_chains)
storage.register_func(plot_pos_likelihoods)
storage.register_func(get_top10_source_chains)
