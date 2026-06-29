"""Single-agent and graph-multi-agent Active Inference on a grid.

This module provides:

* :class:`ActiveGridference` — a generative-model holder for a discrete
  grid-world with the canonical Active Inference matrices ``A``, ``B``,
  ``C``, ``D``, ``E``.
* :func:`actinf_planning_single` — one inference / planning / action step
  for a single :class:`ActiveGridference` instance.
* :func:`actinf_graph` — one inference / planning / action step for every
  node in a `networkx` graph whose nodes carry agent state.

The functions are designed to be used inside cadCAD policy blocks but are
also directly callable (see ``notebooks/`` and ``tests/``).
"""

import itertools

import numpy as np

from blockference.maths import construct_policies, onehot
from blockference.utils import utils as bu

__all__ = [
    "ActiveGridference",
    "actinf_planning_single",
    "actinf_graph",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _move(action_id, env_state, border):
    """Apply a 5-action discrete grid move with wall reflection.

    Action encoding: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=STAY.
    """
    y, x = env_state
    if action_id == 0:  # UP
        y = y - 1 if y > 0 else y
    elif action_id == 1:  # DOWN
        y = y + 1 if y < border else y
    elif action_id == 2:  # LEFT
        x = x - 1 if x > 0 else x
    elif action_id == 3:  # RIGHT
        x = x + 1 if x < border else x
    # action_id == 4 (STAY) is a no-op
    return (y, x)


# ---------------------------------------------------------------------------
# Single-agent step
# ---------------------------------------------------------------------------


def actinf_planning_single(agent, env_state, A, B, C, prior):
    """Run one Active Inference planning step for a single agent.

    Parameters
    ----------
    agent : ActiveGridference
        The agent providing ``n_states``, ``E`` (affordances), ``policy_len``
        and ``border`` (grid boundary index).
    env_state : tuple[int, int]
        Current ``(y, x)`` location on the grid.
    A, B, C : numpy.ndarray
        Generative-model matrices.
    prior : numpy.ndarray
        Prior over hidden states for this timestep.

    Returns
    -------
    dict
        Update payload carrying ``update_prior``, ``update_env``,
        ``update_action``, ``update_inference``, plus the full per-step
        diagnostics: ``update_efe`` (G per policy), ``update_efe_epistemic``,
        ``update_efe_pragmatic``, ``update_q_pi``, ``update_p_u``,
        ``update_obs_idx``.
    """
    policies = construct_policies([agent.n_states], [len(agent.E)], policy_len=agent.policy_len)
    obs_idx = agent.grid.index(env_state)

    qs_current = bu.infer_states(obs_idx, A, prior)
    G, parts = bu.calculate_G_policies_traced(A, B, C, qs_current, policies=policies)
    Q_pi = bu.softmax(-G)
    P_u = bu.compute_prob_actions(agent.E, policies, Q_pi)
    chosen_action = bu.sample(P_u)

    next_prior = B[:, :, chosen_action].dot(qs_current)
    next_env = _move(chosen_action, env_state, agent.border)

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


# ---------------------------------------------------------------------------
# Multi-agent (graph) step
# ---------------------------------------------------------------------------


def actinf_graph(agent_network):
    """Run one Active Inference step over every node in ``agent_network``.

    Each node is expected to expose the keys ``agent`` (an
    :class:`ActiveGridference`), ``env_state``, ``prior``, ``prior_A``,
    ``prior_B``, ``prior_C``.
    """
    agent_updates = []

    for node in agent_network.nodes:
        node_data = agent_network.nodes[node]
        agent = node_data["agent"]

        policies = construct_policies([agent.n_states], [len(agent.E)], policy_len=agent.policy_len)
        obs_idx = agent.grid.index(node_data["env_state"])

        qs_current = bu.infer_states(obs_idx, node_data["prior_A"], node_data["prior"])
        G, parts = bu.calculate_G_policies_traced(
            node_data["prior_A"],
            node_data["prior_B"],
            node_data["prior_C"],
            qs_current,
            policies=policies,
        )
        Q_pi = bu.softmax(-G)
        P_u = bu.compute_prob_actions(agent.E, policies, Q_pi)
        chosen_action = bu.sample(P_u)

        next_prior = node_data["prior_B"][:, :, chosen_action].dot(qs_current)
        next_env = _move(chosen_action, node_data["env_state"], agent.border)

        agent_updates.append(
            {
                "source": node,
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
        )

    return {"agent_updates": agent_updates}


# ---------------------------------------------------------------------------
# Generative-model holder
# ---------------------------------------------------------------------------


class ActiveGridference:
    """Generative model + state container for an Active Inference grid agent.

    The class assembles the canonical discrete-state Active Inference
    matrices for a 2-D grid-world:

    ====  ====================================================================
    Sym.  Meaning
    ====  ====================================================================
    A     Likelihood ``P(o|s)`` — observation given hidden state.
    B     Transition ``P(s_t | s_{t-1}, u_{t-1})`` — controllable dynamics.
    C     Prior preferences over observations.
    D     Prior over hidden states at the first timestep.
    E     Affordances (action labels).
    ====  ====================================================================
    """

    def __init__(
        self,
        grid,
        planning_length: int = 2,
        env_state: tuple = (0, 0),
    ) -> None:
        super().__init__()
        self.A = None
        self.B = None
        self.C = None
        self.D = None
        self.E = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
        self.grid = grid

        self.policy_len = planning_length

        # environment
        self.n_states = len(self.grid)
        self.n_observations = len(self.grid)
        self.border = int(np.sqrt(self.n_states) - 1)

        # active state
        self.prior = self.D
        self.current_action = ""
        self.current_inference = ""
        self.env_state = env_state

        if self.grid is not None:
            self.get_A()
            self.get_B()

    # ------------------------------------------------------------------
    # Generative-model construction
    # ------------------------------------------------------------------

    def get_A(self):
        """Identity likelihood for fully-observed gridworld."""
        self.A = np.eye(self.n_observations, self.n_states)

    def get_B(self):
        """Build the controllable transition tensor B[s', s, a]."""
        self.B = np.zeros((len(self.grid), len(self.grid), len(self.E)))

        for action_id, action_label in enumerate(self.E):
            for curr_state, grid_location in enumerate(self.grid):
                y, x = grid_location

                if action_label == "UP":
                    next_y = y - 1 if y > 0 else y
                    next_x = x
                elif action_label == "DOWN":
                    next_y = y + 1 if y < self.border else y
                    next_x = x
                elif action_label == "LEFT":
                    next_x = x - 1 if x > 0 else x
                    next_y = y
                elif action_label == "RIGHT":
                    next_x = x + 1 if x < self.border else x
                    next_y = y
                elif action_label == "STAY":
                    next_y, next_x = y, x
                else:
                    raise ValueError(f"Unknown action label: {action_label}")

                next_state = self.grid.index((next_y, next_x))
                self.B[next_state, curr_state, action_id] = 1.0

    def get_C(self, preferred_state: tuple):
        """Build a one-hot prior preference vector over observations."""
        self.C = onehot(self.grid.index(preferred_state), self.n_observations)

    def get_D(self, initial_state):
        """Build a one-hot prior over the initial hidden state."""
        self.D = onehot(self.grid.index(initial_state), self.n_states)
        self.prior = self.D

    def get_E(self, actions: list):
        """Override the affordance set."""
        self.E = actions


def make_grid(grid_len: int, grid_dim: int = 2):
    """Convenience: return a list of grid coordinates of length ``grid_len^grid_dim``."""
    return list(itertools.product(range(grid_len), repeat=grid_dim))
