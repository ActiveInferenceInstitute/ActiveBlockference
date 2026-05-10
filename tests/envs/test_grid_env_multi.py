"""Tests for blockference.envs.grid_env_multi."""

import numpy as np

from blockference.envs import TwoMultiGridAgent
from blockference.gridference import ActiveGridference


def _make_two_agent_env():
    grid_len = 3
    agents = [ActiveGridference([(y, x) for y in range(grid_len) for x in range(grid_len)])
              for _ in range(2)]
    init_pos = [0, 8]   # corners (0,0) and (2,2)
    init_obs = [
        [np.eye(9)[0].astype(int), np.eye(9)[8].astype(int)],
        [np.eye(9)[8].astype(int), np.eye(9)[0].astype(int)],
    ]
    return TwoMultiGridAgent(grid_len, agents=agents, init_pos=init_pos, init_obs=init_obs)


def test_two_agent_env_constructs(capsys):
    env = _make_two_agent_env()
    assert env.num_states == 9
    assert len(env.current_state) == 2


def test_two_agent_step_returns_obs(capsys):
    env = _make_two_agent_env()
    # actions are wrapped in a list, mirroring the upstream convention
    new_obs = env.step(actions=[[4], [4]])  # both STAY
    assert len(new_obs) == 2
    # observations remain at the original positions
    assert int(np.argmax(new_obs[0][0])) == 0
    assert int(np.argmax(new_obs[1][0])) == 8
