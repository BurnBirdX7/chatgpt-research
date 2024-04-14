import numpy as np
import pytest

from src import TokenChain


def test_simple_chain():
    chain = TokenChain("source1", 24, 42)

    assert chain.all_likelihoods.dtype == np.float64

    chain.append_end(0.97)
    chain.append_end(0.70)
    chain.skip_end(0.0)

    assert chain.all_likelihoods.dtype == np.float64

    assert len(chain) == 3
    assert chain.begin_skips == 0
    assert chain.end_skips == 1

    assert chain.target_begin_pos == 24
    assert chain.target_end_pos == 24 + 3
    assert chain.source_begin_pos == 42
    assert chain.source_end_pos == 42 + 3

    chain.append_end(0.3)

    assert len(chain) == 4
    assert chain.end_skips == 0


def test_reverse():
    chain = TokenChain("source1", 24, 42)

    chain.skip_end(0)
    chain.skip_end(0.0000001)
    chain.append_end(0.97)
    chain.append_end(0.70)
    chain.skip_end(0.000005)

    def chain_assert():
        assert len(chain) == 5
        assert chain.begin_skips == 2
        assert chain.end_skips == 1

        assert chain.target_begin_pos == 24
        assert chain.target_end_pos == 24 + 5
        assert chain.source_begin_pos == 42
        assert chain.source_end_pos == 42 + 5

    chain_assert()

    r_chain = chain.reverse()

    chain_assert()

    assert chain.all_likelihoods.dtype == r_chain.all_likelihoods.dtype

    assert r_chain.parent is chain

    assert len(r_chain) == 5
    assert r_chain.begin_skips == 1
    assert r_chain.end_skips == 2

    assert r_chain.target_begin_pos == 24 - len(chain) + 1
    assert r_chain.target_end_pos == 24 + 1
    assert r_chain.source_begin_pos == 42 - len(chain) + 1
    assert r_chain.source_end_pos == 42 + 1

    for i in range(len(chain)):
        assert r_chain.all_likelihoods[i] == chain.all_likelihoods[-1 - i]


def test_trim():
    chain = TokenChain("source1", 24, 42)

    chain.skip_end(0.000003)
    chain.skip_end(0.000007)
    chain.append_end(0.99)
    chain.append_end(0.88)
    chain.append_end(0.77)
    chain.skip_end(0.00000001)

    t_chain = chain.trim_copy()

    assert t_chain.parent is chain
    assert len(t_chain) == 3
    assert t_chain.begin_skips == 0
    assert t_chain.end_skips == 0

    assert t_chain.target_begin_pos == chain.target_begin_pos + 2
    assert t_chain.source_begin_pos == chain.source_begin_pos + 2

    tt_chain = t_chain.trim_copy()

    assert len(tt_chain) == 3
    assert t_chain == tt_chain


def test_add():
    chain1 = TokenChain("source1", 24, 42)
    chain2 = TokenChain("source1", 30, 48)

    chain1.skip_end(0.000005)
    chain1.skip_end(0.000003)
    chain1.skip_end(0.000007)
    chain1.append_end(0.99)
    chain1.append_end(0.88)
    chain1.append_end(0.77)

    with pytest.raises(ValueError):
        chain1 + chain2

    chain1.append_end(0.66)

    with pytest.raises(ValueError):
        chain1 + chain2

    chain2.append_end(0.77)

    with pytest.raises(ValueError):
        chain1 + chain2

    chain2.all_likelihoods[0] = 0.66

    chain1 + chain2
    with pytest.raises(ValueError):
        chain2 + chain1

    chain2.append_end(0.55)
    chain2.skip_end(0.00000001)
    chain2.skip_end(0.00000007)

    res_chain = chain1 + chain2

    assert len(res_chain) == len(chain1) + len(chain2) - 1
    assert res_chain.begin_skips == chain1.begin_skips
    assert res_chain.end_skips == chain2.end_skips

    t_chain1 = chain1.trim_copy()
    t_chain2 = chain2.trim_copy()
    assert (t_chain1 + t_chain2) == res_chain.trim_copy()
