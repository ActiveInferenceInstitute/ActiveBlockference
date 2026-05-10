"""Tests for blockference.gridference."""

import numpy as np
import pytest

from blockference.gridference import (
    ActiveGridference,
    _move,
    actinf_planning_single,
    make_grid,
)

# ---------------------------------------------------------------------------
# make_grid
# ---------------------------------------------------------------------------


def test_make_grid_2d():
    grid = make_grid(3, 2)
    assert len(grid) == 9
    assert (0, 0) in grid and (2, 2) in grid


def test_make_grid_3d():
    grid = make_grid(2, 3)
    assert len(grid) == 8
    assert (0, 0, 0) in grid and (1, 1, 1) in grid


# ---------------------------------------------------------------------------
# _move
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "action_id, start, expected",
    [
        (0, (1, 1), (0, 1)),    # UP
        (1, (1, 1), (2, 1)),    # DOWN
        (2, (1, 1), (1, 0)),    # LEFT
        (3, (1, 1), (1, 2)),    # RIGHT
        (4, (1, 1), (1, 1)),    # STAY
        (0, (0, 1), (0, 1)),    # UP at top boundary stays
        (1, (2, 1), (2, 1)),    # DOWN at bottom boundary stays
        (2, (1, 0), (1, 0)),    # LEFT at left boundary stays
        (3, (1, 2), (1, 2)),    # RIGHT at right boundary stays
    ],
)
def test_move_actions(action_id, start, expected):
    assert _move(action_id, start, border=2) == expected


# ---------------------------------------------------------------------------
# ActiveGridference construction
# ---------------------------------------------------------------------------


def test_construction_builds_A_and_B(grid_3x3):
    agent = ActiveGridference(grid_3x3)
    assert agent.A.shape == (9, 9)
    assert np.allclose(agent.A, np.eye(9))
    assert agent.B.shape == (9, 9, 5)


def test_B_columns_are_one_hot(grid_3x3):
    agent = ActiveGridference(grid_3x3)
    # Each column of B[:, :, a] is a probability distribution summing to 1.
    for a in range(agent.B.shape[2]):
        sums = agent.B[:, :, a].sum(axis=0)
        assert np.allclose(sums, 1.0)


def test_get_C_is_one_hot(grid_3x3):
    agent = ActiveGridference(grid_3x3)
    agent.get_C((2, 2))
    assert agent.C.shape == (9,)
    assert agent.C.sum() == pytest.approx(1.0)
    assert agent.C[grid_3x3.index((2, 2))] == pytest.approx(1.0)


def test_get_D_sets_prior(grid_3x3):
    agent = ActiveGridference(grid_3x3)
    agent.get_D((0, 0))
    assert agent.D.sum() == pytest.approx(1.0)
    assert np.allclose(agent.prior, agent.D)


def test_get_E_overrides_actions(grid_3x3):
    agent = ActiveGridference(grid_3x3)
    agent.get_E(["A", "B"])
    assert agent.E == ["A", "B"]


def test_border_is_int(grid_3x3):
    agent = ActiveGridference(grid_3x3)
    assert isinstance(agent.border, int)
    assert agent.border == 2


# ---------------------------------------------------------------------------
# actinf_planning_single
# ---------------------------------------------------------------------------


def test_planning_step_returns_expected_keys(grid_3x3):
    agent = ActiveGridference(grid_3x3, planning_length=1)
    agent.get_C((2, 2))
    agent.get_D((0, 0))

    update = actinf_planning_single(
        agent, agent.env_state, agent.A, agent.B, agent.C, agent.prior
    )

    expected = {
        "update_prior",
        "update_env",
        "update_action",
        "update_inference",
        "update_efe",
        "update_efe_epistemic",
        "update_efe_pragmatic",
        "update_q_pi",
        "update_p_u",
        "update_obs_idx",
    }
    assert set(update) == expected
    assert isinstance(update["update_action"], (int, np.integer))
    assert isinstance(update["update_env"], tuple)
    assert update["update_prior"].shape == (9,)
    # EFE decomposition consistency
    assert np.allclose(
        update["update_efe"],
        update["update_efe_epistemic"] + update["update_efe_pragmatic"],
    )
    # Policy posterior & action marginal are stochastic.
    assert np.isclose(update["update_q_pi"].sum(), 1.0)
    assert np.isclose(update["update_p_u"].sum(), 1.0)
    assert update["update_q_pi"].shape == (5,)  # 5 actions, policy_len=1
    assert update["update_p_u"].shape == (5,)


def test_agent_can_reach_goal(grid_3x3):
    agent = ActiveGridference(grid_3x3, planning_length=2)
    agent.get_C((2, 2))
    agent.get_D((0, 0))

    target = (2, 2)
    for _ in range(50):
        update = actinf_planning_single(
            agent, agent.env_state, agent.A, agent.B, agent.C, agent.prior
        )
        agent.prior = update["update_prior"]
        agent.env_state = update["update_env"]
        if agent.env_state == target:
            break
    assert agent.env_state == target, "agent failed to reach target within budget"
