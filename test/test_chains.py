import numpy as np
import pytest

from src import TokenChain


def test_mapping():
    chain = TokenChain(
        "1",
        8,
        6,
        [0.9, 0.8, 0.7, 0.9, 0.0, 0.0, 0.7, 0.6, 0.8],
        [1, 1, 1, 1, 0, 0, 1, 1, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1],
    )

    s_pos = chain.source_positions()

    assert s_pos == [0, 2, 3, 4, None, None, 5, 6, 8]

    expected_probs = np.array([0.9, 0.8, 0.7, 0.9, 0.0, 0.0, 0.7, 0.6, 0.8], dtype=np.float32)
    for i, y_true in enumerate(expected_probs):
        y_pred = chain.get_target_likelihood(i + 8)
        assert y_pred.dtype == y_true.dtype and y_pred == y_true, f"position: {i} ({i+8})"


def test_simple_chain():
    chain = TokenChain("source1", 24, 42)

    assert chain.likelihoods.dtype == np.float32
    assert len(chain) == 0

    chain.expand_forward(0.97)
    chain.expand_forward(0.70)
    chain.skip_forward(0.0)

    assert chain.likelihoods.dtype == np.float32

    assert len(chain) == 3
    assert chain.target_begin_pos == 24
    assert chain.target_end_pos == 24 + 3
    assert chain.source_begin_pos == 42
    assert chain.source_end_pos == 42 + 3

    chain.expand_forward(0.3)

    assert len(chain) == 4
    assert chain.target_begin_pos == 24
    assert chain.target_end_pos == 24 + 4
    assert chain.source_begin_pos == 42
    assert chain.source_end_pos == 42 + 4

    chain.expand_backward(0.88)

    assert len(chain) == 5
    assert chain.target_begin_pos == 24 - 1
    assert chain.target_end_pos == 24 + 4
    assert chain.source_begin_pos == 42 - 1
    assert chain.source_end_pos == 42 + 4


def test_trim():
    chain = TokenChain("source1", 24, 42)

    chain.skip_forward(0.000003)
    chain.skip_forward(0.000007)
    chain.expand_forward(0.99)
    chain.expand_forward(0.88)
    chain.expand_forward(0.77)
    chain.skip_forward(0.00000001)

    t_chain = chain.trim_copy()

    assert t_chain.parent is chain
    assert len(t_chain) == 3
    assert t_chain.target_mask.sum() == 3

    assert t_chain.target_begin_pos == chain.target_begin_pos + 2
    assert t_chain.source_begin_pos == chain.source_begin_pos + 2

    tt_chain = t_chain.trim_copy()

    assert len(tt_chain) == 3
    assert t_chain == tt_chain


def test_add():
    chain1 = TokenChain("source1", 24, 42)
    chain2 = TokenChain("source1", 30, 48)

    chain1.skip_forward(0.000005)
    chain1.skip_forward(0.000003)
    chain1.skip_forward(0.000007)
    chain1.expand_forward(0.99)
    chain1.expand_forward(0.88)
    chain1.expand_forward(0.77)

    with pytest.raises(ValueError):
        chain1 + chain2

    chain1.expand_forward(0.66)

    with pytest.raises(ValueError):
        chain1 + chain2

    chain2.expand_forward(0.77)

    with pytest.raises(ValueError):
        chain1 + chain2

    chain2.likelihoods[0] = 0.66  # Make likelihoods match in the middle

    chain1 + chain2
    with pytest.raises(ValueError):
        chain2 + chain1

    chain2.expand_forward(0.55)
    chain2.skip_forward(0.00000001)
    chain2.skip_forward(0.00000007)

    res_chain = chain1 + chain2

    assert len(res_chain) == len(chain1) + len(chain2) - 1
    assert np.array_equal(res_chain.target_mask[:3], [False, False, False])
    assert np.array_equal(res_chain.target_mask[-3:], [True, False, False])

    t_chain1 = chain1.trim_copy()
    t_chain2 = chain2.trim_copy()
    assert (t_chain1 + t_chain2) == res_chain.trim_copy()
