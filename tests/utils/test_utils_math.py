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
    B[:, :, 0] = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])  # shift down
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
