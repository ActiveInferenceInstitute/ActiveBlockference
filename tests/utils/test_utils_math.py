"""Tests for blockference.utils.utils — the AIF math helpers."""

import numpy as np
import pytest

from blockference.utils import utils as u


def test_entropy_uniform_is_log_n():
    n = 4
    A = np.full((n, n), 1.0 / n)
    H = u.entropy(A)
    assert H.shape == (n,)
    assert np.allclose(H, np.log(n))


def test_entropy_one_hot_is_zero():
    n = 3
    A = np.eye(n)
    H = u.entropy(A)
    assert np.allclose(H, 0.0)


def test_kl_divergence_self_is_zero():
    p = np.array([0.2, 0.3, 0.5])
    assert u.kl_divergence(p, p) == pytest.approx(0.0, abs=1e-9)


def test_infer_states_recovers_observation_for_identity_A():
    n = 4
    A = np.eye(n)
    prior = np.full(n, 1.0 / n)
    qs = u.infer_states(observation_index=2, A=A, prior=prior)
    assert qs.shape == (n,)
    assert np.argmax(qs) == 2


def test_get_expected_states_propagates_through_B():
    n = 3
    qs = np.array([1.0, 0.0, 0.0])
    B = np.zeros((n, n, 1))
    B[:, :, 0] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]])  # shift down; bounded at the end
    qs_next = u.get_expected_states(B, qs, 0)
    assert np.allclose(qs_next, [0, 1, 0])


def test_compute_prob_actions_marginalises():
    actions = ["A", "B"]
    # Two policies, each picking one action at t=0:
    policies = [np.array([[0]]), np.array([[1]])]
    Q_pi = np.array([0.7, 0.3])
    P_u = u.compute_prob_actions(actions, policies, Q_pi)
    assert np.allclose(P_u, [0.7, 0.3])
    assert P_u.sum() == pytest.approx(1.0)


def test_calculate_G_policies_shapes():
    n = 4
    A = np.eye(n)
    B = np.stack([np.eye(n)] * 2, axis=-1)
    C = np.array([0.0, 0.0, 0.0, 1.0])
    qs_current = np.full(n, 1.0 / n)
    policies = [np.array([[0]]), np.array([[1]])]
    G = u.calculate_G_policies(A, B, C, qs_current, policies)
    assert G.shape == (2,)


def test_calculate_G_single_step_efe():
    # Identity likelihood => ambiguity zero; a switch action moves away from C.
    A = np.eye(2)
    B = np.zeros((2, 2, 2))
    B[:, :, 0] = np.eye(2)
    B[:, :, 1] = np.array([[0.0, 1.0], [1.0, 0.0]])
    C = np.array([1.0, 0.0])
    qs = np.array([1.0, 0.0])
    G = u.calculate_G(A, B, C, qs, ["stay", "switch"])
    assert G.shape == (2,)
    assert G[0] == pytest.approx(0.0, abs=1e-9)
    assert G[1] > G[0]


def test_sample_without_rng_uses_global_stream():
    np.random.seed(7)
    first = u.sample(np.array([0.5, 0.5]))
    np.random.seed(7)
    second = u.sample(np.array([0.5, 0.5]))
    assert first == second


@pytest.fixture
def preserved_global_rng():
    state = np.random.get_state()
    try:
        yield
    finally:
        np.random.set_state(state)


def _assert_global_state_equal(expected):
    actual = np.random.get_state()
    assert actual[0] == expected[0]
    assert np.array_equal(actual[1], expected[1])
    assert actual[2:] == expected[2:]


def test_extreme_finite_weights_normalize_per_slice_without_overflow():
    largest = np.finfo(float).max
    smallest = np.nextafter(0.0, 1.0)
    weights = np.array([[largest, largest / 2], [0.0, 0.0], [smallest, 2 * smallest]])
    expected = np.array([[2 / 3, 1 / 3], [0.5, 0.5], [1 / 3, 2 / 3]])
    original = weights.copy()
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        rows = u.norm_dist(weights, axis=1)
        columns = u.norm_dist(weights.T, axis=0)
        tensor = u.norm_dist(np.stack([weights, weights]), axis=-1)
    assert np.isfinite(rows).all()
    assert np.allclose(rows, expected)
    assert np.allclose(rows.sum(axis=1), 1.0)
    assert np.allclose(columns, expected.T)
    assert np.allclose(tensor, np.stack([expected, expected]))
    assert np.array_equal(weights, original)


def test_extreme_sampling_matches_owned_generator_without_touching_global(preserved_global_rng):
    largest = np.finfo(float).max
    rng = np.random.default_rng(31)
    expected_rng = np.random.default_rng(31)
    global_state = np.random.get_state()
    actual = [u.sample(np.array([largest, largest / 2]), rng=rng) for _ in range(64)]
    expected = expected_rng.choice(2, size=64, p=[2 / 3, 1 / 3])
    assert np.array_equal(actual, expected)
    assert rng.bit_generator.state == expected_rng.bit_generator.state
    _assert_global_state_equal(global_state)


def test_extreme_sampling_preserves_legacy_global_stream(preserved_global_rng):
    np.random.seed(23)
    expected = np.random.choice(2, size=64, p=[2 / 3, 1 / 3])
    expected_state = np.random.get_state()
    np.random.seed(23)
    largest = np.finfo(float).max
    actual = [u.sample(np.array([largest, largest / 2])) for _ in range(64)]
    assert np.array_equal(actual, expected)
    _assert_global_state_equal(expected_state)


@pytest.mark.parametrize("weights", [[], [0, 0], [-1, 2], [np.nan, 1], [np.inf, 1], [[1, 2]]])
def test_sampling_refuses_invalid_mass_before_consuming_any_rng(weights, preserved_global_rng):
    rng = np.random.default_rng(41)
    owned_state = rng.bit_generator.state
    global_state = np.random.get_state()
    with pytest.raises(ValueError):
        u.sample(weights, rng=rng)
    with pytest.raises(ValueError):
        u.sample(weights)
    assert rng.bit_generator.state == owned_state
    _assert_global_state_equal(global_state)


def test_retained_conditional_ambiguity_is_not_predictive_entropy():
    likelihood = np.eye(2)
    belief = np.array([0.5, 0.5])
    transitions = np.eye(2)[:, :, None]
    total, parts = u.calculate_G_policies_traced(
        likelihood, transitions, belief, belief, [np.array([[0]])],
    )
    predictive = likelihood @ belief
    predictive_entropy = -float(np.sum(predictive * np.log(predictive)))
    assert np.allclose(parts["epistemic"], [0.0])
    assert np.allclose(parts["pragmatic"], [0.0])
    assert np.allclose(total, [0.0])
    assert predictive_entropy == pytest.approx(np.log(2))
