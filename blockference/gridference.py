"""Discrete-state Active Inference on a square grid."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any, TypedDict, cast

import numpy as np

from blockference.actions import DEFAULT_AFFORDANCES, SUPPORTED_AFFORDANCES, validate_affordances
from blockference.maths import construct_policies, onehot
from blockference.utils import utils as bu

__all__ = [
    "DEFAULT_AFFORDANCES",
    "SUPPORTED_AFFORDANCES",
    "ActiveGridference",
    "AgentUpdate",
    "actinf_graph",
    "actinf_planning_single",
    "make_grid",
]


class AgentUpdate(TypedDict, total=False):
    """CadCAD-compatible update envelope shared by all simulation backends."""

    source: Any
    update_prior: np.ndarray
    update_env: tuple[int, int]
    update_action: int
    update_inference: np.ndarray
    update_efe: np.ndarray
    update_efe_epistemic: np.ndarray
    update_efe_pragmatic: np.ndarray
    update_q_pi: np.ndarray
    update_p_u: np.ndarray
    update_obs_idx: int


def _validate_affordances(affordances: Sequence[str]) -> list[str]:
    return list(validate_affordances(affordances))


def _move(
    action_id: int,
    env_state: tuple[int, int],
    border: int,
    affordances: Sequence[str] = DEFAULT_AFFORDANCES,
) -> tuple[int, int]:
    """Apply one validated action using the package coordinate convention."""
    if isinstance(action_id, (bool, np.bool_)) or not isinstance(action_id, (int, np.integer)):
        raise TypeError("action_id must be an integer")
    if (
        isinstance(border, (bool, np.bool_))
        or not isinstance(border, (int, np.integer))
        or border < 0
    ):
        raise ValueError("border must be a non-negative integer")
    if not isinstance(env_state, (tuple, list)) or len(env_state) != 2:
        raise ValueError("env_state must be a (y, x) coordinate")
    if any(
        isinstance(component, (bool, np.bool_)) or not isinstance(component, (int, np.integer))
        for component in env_state
    ):
        raise TypeError("env_state coordinates must be integers")
    y, x = (int(env_state[0]), int(env_state[1]))
    if not (0 <= y <= border and 0 <= x <= border):
        raise ValueError(
            f"env_state {(y, x)} is outside a {int(border) + 1}x{int(border) + 1} grid"
        )
    actions = _validate_affordances(affordances)
    if not 0 <= int(action_id) < len(actions):
        raise IndexError(f"action_id {action_id} out of range [0, {len(actions)})")
    label = actions[int(action_id)]
    if label == "UP":
        y = max(0, y - 1)
    elif label == "DOWN":
        y = min(int(border), y + 1)
    elif label == "LEFT":
        x = max(0, x - 1)
    elif label == "RIGHT":
        x = min(int(border), x + 1)
    return y, x


def _validate_square_grid(grid: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    values = [tuple(location) for location in grid]
    if not values or any(len(location) != 2 for location in values):
        raise ValueError("grid must be a non-empty sequence of two-dimensional coordinates")
    if len(set(values)) != len(values):
        raise ValueError("grid coordinates must be unique")
    side = int(round(len(values) ** 0.5))
    expected = set(itertools.product(range(side), repeat=2))
    if side * side != len(values) or set(values) != expected:
        raise ValueError("ActiveGridference requires a complete square grid starting at (0, 0)")
    return cast(list[tuple[int, int]], values)


def _coordinate(value, name: str) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-dimensional integer coordinate")
    if any(
        isinstance(component, (bool, np.bool_)) or not isinstance(component, (int, np.integer))
        for component in value
    ):
        raise TypeError(f"{name} coordinates must be integers")
    return int(value[0]), int(value[1])


def _step_agent(agent, prior, A, B, C, env_state, grid, *, rng=None) -> AgentUpdate:
    policies = construct_policies(
        [agent.n_states],
        [len(agent.E)],
        policy_len=agent.policy_len,
        max_policies=getattr(agent, "max_policies", 100_000),
    )
    obs_idx = grid.index(tuple(env_state))
    qs_current = bu.infer_states(obs_idx, A, prior)
    G, parts = bu.calculate_G_policies_traced(A, B, C, qs_current, policies=policies)
    Q_pi = bu.softmax(-G)
    P_u = bu.compute_prob_actions(agent.E, policies, Q_pi)
    chosen_action = bu.sample(P_u, rng=rng)
    next_prior = B[:, :, chosen_action].dot(qs_current)
    next_env = _move(chosen_action, tuple(env_state), agent.border, agent.E)
    return {
        "update_prior": next_prior,
        "update_env": next_env,
        "update_action": int(chosen_action),
        "update_inference": qs_current,
        "update_efe": G,
        "update_efe_epistemic": parts["epistemic"],
        "update_efe_pragmatic": parts["pragmatic"],
        "update_q_pi": Q_pi,
        "update_p_u": P_u,
        "update_obs_idx": int(obs_idx),
    }


def actinf_planning_single(
    agent, env_state, A, B, C, prior, *, rng=None
) -> AgentUpdate:
    """Run one inference, planning, sampling, and prior-propagation step."""
    return _step_agent(agent, prior, A, B, C, env_state, agent.grid, rng=rng)


def actinf_graph(agent_network, *, rng=None) -> dict[str, list[AgentUpdate]]:
    """Run one independent Active Inference step for every graph node."""
    updates = []
    for node in agent_network.nodes:
        data = agent_network.nodes[node]
        update = _step_agent(
            data["agent"],
            data["prior"],
            data["prior_A"],
            data["prior_B"],
            data["prior_C"],
            data["env_state"],
            data["agent"].grid,
            rng=rng,
        )
        update["source"] = node
        updates.append(update)
    return {"agent_updates": updates}


class ActiveGridference:
    """Generative model and mutable state for a square-grid agent."""

    def __init__(
        self,
        grid,
        planning_length: int = 2,
        env_state: tuple[int, int] = (0, 0),
        affordances: Sequence[str] = DEFAULT_AFFORDANCES,
        max_policies: int = 100_000,
    ) -> None:
        if (
            isinstance(planning_length, (bool, np.bool_))
            or not isinstance(planning_length, (int, np.integer))
            or planning_length < 1
        ):
            raise ValueError("planning_length must be a positive integer")
        self.grid = _validate_square_grid(grid)
        self.E = _validate_affordances(affordances)
        if (
            isinstance(max_policies, (bool, np.bool_))
            or not isinstance(max_policies, (int, np.integer))
            or max_policies < 1
        ):
            raise ValueError("max_policies must be a positive integer")
        self.max_policies = int(max_policies)
        self.policy_len = int(planning_length)
        self.n_states = len(self.grid)
        self.n_observations = len(self.grid)
        self.border = int(round(self.n_states**0.5)) - 1
        self.env_state = _coordinate(env_state, "env_state")
        if self.env_state not in self.grid:
            raise ValueError(f"env_state {self.env_state!r} is not present in grid")
        self.A: np.ndarray = np.eye(self.n_observations, self.n_states)
        self.B: np.ndarray = self._build_B()
        self.C: np.ndarray | None = None
        self.D: np.ndarray | None = None
        self.prior: np.ndarray | None = None
        self.current_action: int | None = None
        self.current_inference: np.ndarray | None = None

    def _build_B(self) -> np.ndarray:
        B = np.zeros((self.n_states, self.n_states, len(self.E)), dtype=float)
        for action_id in range(len(self.E)):
            for current_index, location in enumerate(self.grid):
                next_location = _move(action_id, location, self.border, self.E)
                B[self.grid.index(next_location), current_index, action_id] = 1.0
        return B

    def get_A(self) -> np.ndarray:
        """Rebuild and return the identity likelihood matrix."""
        self.A = np.eye(self.n_observations, self.n_states)
        return self.A

    def get_B(self) -> np.ndarray:
        """Rebuild and return the affordance-specific transition tensor."""
        self.B = self._build_B()
        return self.B

    def get_C(self, preferred_state: tuple[int, int]) -> np.ndarray:
        """Set and return a one-hot preference over observations."""
        coordinate = _coordinate(preferred_state, "preferred_state")
        if coordinate not in self.grid:
            raise ValueError(f"preferred_state {preferred_state!r} is not present in grid")
        self.C = onehot(self.grid.index(coordinate), self.n_observations)
        return self.C

    def get_D(self, initial_state: tuple[int, int]) -> np.ndarray:
        """Set and return a one-hot prior over hidden states."""
        coordinate = _coordinate(initial_state, "initial_state")
        if coordinate not in self.grid:
            raise ValueError(f"initial_state {initial_state!r} is not present in grid")
        self.D = onehot(self.grid.index(coordinate), self.n_states)
        self.prior = self.D.copy()
        self.env_state = coordinate
        return self.D

    def get_E(self, actions: Sequence[str]) -> list[str]:
        """Set affordances and rebuild the transition tensor."""
        self.E = _validate_affordances(actions)
        self.B = self._build_B()
        return self.E


def make_grid(grid_len: int, grid_dim: int = 2) -> list[tuple[int, ...]]:
    """Return Cartesian grid coordinates for positive dimensions."""
    if (
        isinstance(grid_len, (bool, np.bool_))
        or not isinstance(grid_len, (int, np.integer))
        or grid_len < 1
    ):
        raise ValueError("grid_len must be a positive integer")
    if (
        isinstance(grid_dim, (bool, np.bool_))
        or not isinstance(grid_dim, (int, np.integer))
        or grid_dim < 1
    ):
        raise ValueError("grid_dim must be a positive integer")
    return list(itertools.product(range(int(grid_len)), repeat=int(grid_dim)))  # type: ignore[return-value]
