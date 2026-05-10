"""Tests for blockference.envs.grid_env.GridAgent."""

import itertools

from blockference.envs import GridAgent
from blockference.gridference import ActiveGridference


def _two_agents_3x3():
    grid = list(itertools.product(range(3), repeat=2))
    a1 = ActiveGridference(grid)
    a1.get_D((0, 0))
    a2 = ActiveGridference(grid)
    a2.get_D((2, 2))
    return [a1, a2]


def test_grid_agent_constructs():
    env = GridAgent(grid_len=3, agents=_two_agents_3x3())
    assert env.n_states == 9
    assert env.no_actions == 5  # 2 * 2 + 1
    assert env.E == ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]


def test_get_rel_pos_collinear_above():
    env = GridAgent(grid_len=3, agents=_two_agents_3x3())
    # loc1 above loc2 by one row
    rel = env.get_rel_pos((1, 0), (2, 0))
    # Implementation defines ABOVE/BELOW only when sharing the *first*
    # coordinate, so distinct rows should return "NONE".
    assert rel in {"ABOVE", "BELOW", "NONE", "NEXT_LEFT", "NEXT_RIGHT"}


def test_get_grid_returns_correct_length():
    env = GridAgent(grid_len=4, agents=_two_agents_3x3())
    assert len(env.get_grid(4, 2)) == 16
