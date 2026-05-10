"""Tests for blockference.maths — the self-contained AIF primitives."""

import numpy as np
import pytest

from blockference.maths import (
    MIN_PROB,
    construct_policies,
    log_stable,
    norm_dist,
    onehot,
    sample,
    softmax,
)

# ---------------------------------------------------------------------------
# log_stable
# ---------------------------------------------------------------------------


def test_log_stable_clips_zero():
    assert np.isfinite(log_stable(np.array([0.0, 1.0]))).all()


def test_log_stable_matches_log_for_positive_values():
    x = np.array([0.1, 0.5, 0.9])
    assert np.allclose(log_stable(x), np.log(x))


def test_log_stable_floor_value():
    assert log_stable(np.array([0.0])) == pytest.approx(np.log(MIN_PROB))


# ---------------------------------------------------------------------------
# softmax
# ---------------------------------------------------------------------------


def test_softmax_sums_to_one():
    out = softmax(np.array([1.0, 2.0, 3.0]))
    assert out.sum() == pytest.approx(1.0)


def test_softmax_handles_large_values():
    out = softmax(np.array([1000.0, 1001.0]))
    assert np.isfinite(out).all()
    assert out.sum() == pytest.approx(1.0)


def test_softmax_invariant_to_constant_shift():
    x = np.array([1.0, 2.0, 3.0])
    a = softmax(x)
    b = softmax(x + 100)
    assert np.allclose(a, b)


def test_softmax_2d_axis():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = softmax(x, axis=1)
    assert np.allclose(out.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# norm_dist
# ---------------------------------------------------------------------------


def test_norm_dist_normalises_columns():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = norm_dist(x, axis=0)
    assert np.allclose(out.sum(axis=0), 1.0)


def test_norm_dist_uniform_for_zero_sum():
    x = np.zeros(4)
    out = norm_dist(x, axis=0)
    assert np.allclose(out, np.full(4, 0.25))


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------


def test_sample_returns_int_in_range():
    np.random.seed(42)
    p = np.array([0.1, 0.3, 0.6])
    for _ in range(20):
        idx = sample(p)
        assert isinstance(idx, int)
        assert 0 <= idx < 3


def test_sample_deterministic_with_seed():
    np.random.seed(123)
    p = np.array([0.2, 0.8])
    a = [sample(p) for _ in range(5)]
    np.random.seed(123)
    b = [sample(p) for _ in range(5)]
    assert a == b


def test_sample_renormalises_unnormalised_input():
    p = np.array([2.0, 2.0, 4.0])  # sums to 8, not 1
    np.random.seed(0)
    idx = sample(p)
    assert 0 <= idx < 3


def test_sample_rejects_nan():
    with pytest.raises(ValueError, match="non-finite"):
        sample(np.array([0.5, float("nan")]))


# ---------------------------------------------------------------------------
# onehot
# ---------------------------------------------------------------------------


def test_onehot_basic():
    v = onehot(2, 5)
    assert v.shape == (5,)
    assert v.sum() == pytest.approx(1.0)
    assert v[2] == 1.0


def test_onehot_rejects_out_of_range():
    with pytest.raises(IndexError):
        onehot(5, 5)
    with pytest.raises(IndexError):
        onehot(-1, 5)


# ---------------------------------------------------------------------------
# construct_policies
# ---------------------------------------------------------------------------


def test_construct_policies_count():
    # 5 actions, length-2 horizon → 5*5 = 25 policies
    ps = construct_policies([9], [5], policy_len=2)
    assert len(ps) == 25
    for p in ps:
        assert p.shape == (2, 1)


def test_construct_policies_all_unique():
    ps = construct_policies([4], [3], policy_len=3)
    assert len(ps) == 27
    seen = {tuple(p.flatten().tolist()) for p in ps}
    assert len(seen) == 27


def test_construct_policies_rejects_zero_horizon():
    with pytest.raises(ValueError, match="policy_len"):
        construct_policies([9], [5], policy_len=0)


def test_construct_policies_requires_num_controls():
    with pytest.raises(ValueError, match="num_controls"):
        construct_policies([9])
