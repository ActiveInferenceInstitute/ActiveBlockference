"""Active Inference math + plotting helpers.

The math primitives come from :mod:`blockference.maths` (self-contained
NumPy, no pymdp dependency). This module adds the higher-level Active
Inference operations — state inference, expected free energy — plus
matplotlib/seaborn convenience plotters used by the historical
notebooks.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from blockference.maths import (
    log_stable,
    norm_dist,
    onehot,
    sample,
    softmax,
)

__all__ = [
    "calculate_G",
    "calculate_G_policies",
    "calculate_G_policies_traced",
    "compute_prob_actions",
    "entropy",
    "get_expected_observations",
    "get_expected_states",
    "infer_states",
    "kl_divergence",
    "log_stable",
    "norm_dist",
    "onehot",
    "plot_beliefs",
    "plot_grid",
    "plot_likelihood",
    "plot_point_on_grid",
    "sample",
    "softmax",
]


# ---------------------------------------------------------------------------
# Plotting helpers (notebook-facing)
# ---------------------------------------------------------------------------


def plot_likelihood(matrix, xlabels=None, ylabels=None, title_str="Likelihood distribution (A)"):
    """Render a 2-D likelihood matrix as a heatmap."""
    if xlabels is None:
        xlabels = list(range(matrix.shape[1]))
    if ylabels is None:
        ylabels = list(range(matrix.shape[0]))
    if not np.isclose(matrix.sum(axis=0), 1.0).all():
        raise ValueError("Distribution not column-normalized (matrix.sum(axis=0) must be 1).")
    plt.figure(figsize=(6, 6))
    sns.heatmap(
        matrix,
        xticklabels=xlabels,
        yticklabels=ylabels,
        cmap="gray",
        cbar=False,
        vmin=0.0,
        vmax=1.0,
    )
    plt.title(title_str)
    plt.show()


def plot_grid(grid_locations, num_x=3, num_y=3):
    """Plot the grid heatmap with each (x, y) labelled by linear index."""
    grid_heatmap = np.zeros((num_x, num_y))
    for linear_idx, (y, x) in enumerate(grid_locations):
        grid_heatmap[y, x] = linear_idx
    sns.set(font_scale=1.5)
    sns.heatmap(grid_heatmap, annot=True, cbar=False, fmt=".0f", cmap="crest")


def plot_point_on_grid(state_vector, grid_locations):
    """Highlight the agent's current cell on a 3×3 grid."""
    state_index = int(np.argmax(state_vector))
    y, x = grid_locations[state_index]
    grid_heatmap = np.zeros((3, 3))
    grid_heatmap[y, x] = 1.0
    sns.heatmap(grid_heatmap, cbar=False, fmt=".0f")


def plot_beliefs(belief_dist, title_str: str = ""):
    """Bar-chart a 1-D categorical / belief distribution."""
    if not np.isclose(belief_dist.sum(), 1.0):
        raise ValueError("Distribution not normalized.")
    plt.figure()
    plt.grid(zorder=0)
    plt.bar(range(belief_dist.shape[0]), belief_dist, color="r", zorder=3)
    plt.xticks(range(belief_dist.shape[0]))
    plt.title(title_str)
    plt.show()


# ---------------------------------------------------------------------------
# Active Inference operations
# ---------------------------------------------------------------------------


def get_expected_states(B: np.ndarray, qs_current: np.ndarray, action: int) -> np.ndarray:
    """Compute ``q(s_{t+1} | a)`` by propagating ``qs_current`` through B."""
    return B[:, :, int(action)].dot(qs_current)


def get_expected_observations(A: np.ndarray, qs_u: np.ndarray) -> np.ndarray:
    """Compute ``q(o_{t+1} | a)`` from the A matrix and expected states."""
    return A.dot(qs_u)


def entropy(A: np.ndarray) -> np.ndarray:
    """Per-column entropy of an observation model ``P(o|s)``."""
    return -(A * log_stable(A)).sum(axis=0)


def kl_divergence(qo_u: np.ndarray, C: np.ndarray) -> float:
    """KL between predicted observations and prior preferences."""
    return float((log_stable(qo_u) - log_stable(C)).dot(qo_u))


def infer_states(observation_index: int, A: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Fixed-point state-inference: ``softmax(log A[o, :] + log prior)``."""
    log_likelihood = log_stable(A[int(observation_index), :])
    log_prior = log_stable(prior)
    return softmax(log_likelihood + log_prior)


def calculate_G(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    qs_current: np.ndarray,
    actions,
) -> np.ndarray:
    """Single-step EFE per action."""
    G = np.zeros(len(actions))
    H_A = entropy(A)
    for action_i in range(len(actions)):
        qs_u = get_expected_states(B, qs_current, action_i)
        qo_u = get_expected_observations(A, qs_u)
        G[action_i] = H_A.dot(qs_u) + kl_divergence(qo_u, C)
    return G


def calculate_G_policies(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    qs_current: np.ndarray,
    policies,
) -> np.ndarray:
    """EFE per policy summed over the policy's temporal horizon."""
    G = np.zeros(len(policies))
    H_A = entropy(A)
    for policy_id, policy in enumerate(policies):
        t_horizon = policy.shape[0]
        qs_pi_t = qs_current
        G_pi = 0.0
        for t in range(t_horizon):
            action = int(policy[t, 0])
            qs_prev = qs_current if t == 0 else qs_pi_t
            qs_pi_t = get_expected_states(B, qs_prev, action)
            qo_pi_t = get_expected_observations(A, qs_pi_t)
            G_pi += H_A.dot(qs_pi_t) + kl_divergence(qo_pi_t, C)
        G[policy_id] = G_pi
    return G


def calculate_G_policies_traced(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    qs_current: np.ndarray,
    policies,
) -> tuple[np.ndarray, dict]:
    """Same as :func:`calculate_G_policies` but also returns intermediate quantities.

    The returned dict has keys ``epistemic`` (per-policy ambiguity term) and
    ``pragmatic`` (per-policy KL-to-C term), so callers can persist the
    decomposition for analysis.
    """
    G = np.zeros(len(policies))
    epistemic = np.zeros(len(policies))
    pragmatic = np.zeros(len(policies))
    H_A = entropy(A)
    for policy_id, policy in enumerate(policies):
        t_horizon = policy.shape[0]
        qs_pi_t = qs_current
        amb = 0.0
        risk = 0.0
        for t in range(t_horizon):
            action = int(policy[t, 0])
            qs_prev = qs_current if t == 0 else qs_pi_t
            qs_pi_t = get_expected_states(B, qs_prev, action)
            qo_pi_t = get_expected_observations(A, qs_pi_t)
            amb += float(H_A.dot(qs_pi_t))
            risk += kl_divergence(qo_pi_t, C)
        epistemic[policy_id] = amb
        pragmatic[policy_id] = risk
        G[policy_id] = amb + risk
    return G, {"epistemic": epistemic, "pragmatic": pragmatic}


def compute_prob_actions(actions, policies, Q_pi: np.ndarray) -> np.ndarray:
    """Marginalise the policy posterior down to per-action probability."""
    P_u = np.zeros(len(actions))
    for policy_id, policy in enumerate(policies):
        P_u[int(policy[0, 0])] += Q_pi[policy_id]
    return norm_dist(P_u)


# Convenience stub kept for any caller that imported the old typed alias.
def _noop_optional_int_doc(_: int | None = None) -> None:  # pragma: no cover
    """Placeholder kept to preserve module signatures in older imports."""
    return None
