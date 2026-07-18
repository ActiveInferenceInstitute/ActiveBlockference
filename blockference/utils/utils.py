"""Higher-level Active Inference operations and notebook plotting helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from blockference.maths import log_stable, norm_dist, onehot, sample, softmax

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


def _probability_vector(value, *, name: str, expected_size: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if expected_size is not None and array.size != expected_size:
        raise ValueError(f"{name} must have length {expected_size}, got {array.size}")
    if array.size == 0 or not np.isfinite(array).all() or (array < 0).any():
        raise ValueError(f"{name} must be a non-empty finite non-negative vector")
    if not np.isclose(array.sum(), 1.0, atol=1e-8):
        raise ValueError(f"{name} must sum to 1")
    return array


def _integer_index(value, *, name: str) -> int:
    """Return an integer index without accepting booleans or fractional values."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def plot_likelihood(matrix, xlabels=None, ylabels=None, title_str="Likelihood distribution (A)") -> Any:
    """Render a column-normalised likelihood matrix."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or (matrix < 0).any() or not np.isfinite(matrix).all():
        raise ValueError("matrix must be a finite non-negative two-dimensional array")
    if not np.allclose(matrix.sum(axis=0), 1.0):
        raise ValueError("matrix must be column-normalised")
    xlabels = list(range(matrix.shape[1])) if xlabels is None else xlabels
    ylabels = list(range(matrix.shape[0])) if ylabels is None else ylabels
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(matrix, xticklabels=xlabels, yticklabels=ylabels, cmap="gray", cbar=False, ax=ax)
    ax.set_title(title_str)
    fig.tight_layout()
    return fig


def plot_grid(grid_locations, num_x=None, num_y=None) -> Any:
    """Render a labelled two-dimensional coordinate grid."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    locations = [tuple(location) for location in grid_locations]
    if not locations or any(len(location) != 2 for location in locations):
        raise ValueError("grid_locations must contain two-dimensional coordinates")
    max_y = max(y for y, _ in locations) + 1
    max_x = max(x for _, x in locations) + 1
    rows = max_y if num_y is None else int(num_y)
    cols = max_x if num_x is None else int(num_x)
    if rows < max_y or cols < max_x:
        raise ValueError("grid dimensions are smaller than the supplied coordinates")
    grid_heatmap = np.full((rows, cols), np.nan)
    for linear_idx, (y, x) in enumerate(locations):
        grid_heatmap[y, x] = linear_idx
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(grid_heatmap, annot=True, cbar=False, fmt=".0f", cmap="crest", ax=ax)
    return fig


def plot_point_on_grid(state_vector, grid_locations) -> Any:
    """Render the most probable state on its coordinate grid."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    state = _probability_vector(state_vector, name="state_vector")
    locations = [tuple(location) for location in grid_locations]
    if len(locations) != state.size:
        raise ValueError("state_vector length must equal the number of grid locations")
    rows = max(y for y, _ in locations) + 1
    cols = max(x for _, x in locations) + 1
    grid_heatmap = np.zeros((rows, cols), dtype=float)
    y, x = locations[int(np.argmax(state))]
    grid_heatmap[y, x] = 1.0
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(grid_heatmap, cbar=False, fmt=".0f", ax=ax)
    return fig


def plot_beliefs(belief_dist, title_str: str = "") -> Any:
    """Render a one-dimensional belief distribution."""
    import matplotlib.pyplot as plt

    belief = _probability_vector(belief_dist, name="belief_dist")
    fig, ax = plt.subplots()
    ax.grid(zorder=0)
    ax.bar(range(belief.size), belief, color="r", zorder=3)
    ax.set_xticks(range(belief.size))
    ax.set_title(title_str)
    return fig


def get_expected_states(B: np.ndarray, qs_current: np.ndarray, action: int) -> np.ndarray:
    """Propagate a state posterior through one transition slice."""
    B = np.asarray(B, dtype=float)
    qs = _probability_vector(qs_current, name="qs_current")
    if B.ndim != 3 or B.shape[0] != B.shape[1] or B.shape[1] != qs.size:
        raise ValueError("B must have shape (n_states, n_states, n_actions)")
    if not np.isfinite(B).all() or (B < 0).any() or not np.allclose(B.sum(axis=0), 1.0, atol=1e-8):
        raise ValueError("B must be finite, non-negative, and column-stochastic")
    action_index = _integer_index(action, name="action")
    if not 0 <= action_index < B.shape[2]:
        raise IndexError(f"action {action} is out of range")
    result = B[:, :, action_index].dot(qs)
    return norm_dist(result)


def get_expected_observations(A: np.ndarray, qs_u: np.ndarray) -> np.ndarray:
    """Project an expected state posterior through the likelihood matrix."""
    A = np.asarray(A, dtype=float)
    qs = _probability_vector(qs_u, name="qs_u")
    if A.ndim != 2 or A.shape[1] != qs.size or (A < 0).any() or not np.isfinite(A).all():
        raise ValueError("A must be a finite non-negative matrix compatible with qs_u")
    if not np.allclose(A.sum(axis=0), 1.0, atol=1e-8):
        raise ValueError("A must be column-normalised")
    return norm_dist(A.dot(qs), axis=0)


def entropy(A: np.ndarray) -> np.ndarray:
    """Return per-column entropy for a likelihood matrix."""
    matrix = np.asarray(A, dtype=float)
    if matrix.ndim != 2 or (matrix < 0).any() or not np.isfinite(matrix).all():
        raise ValueError("A must be a finite non-negative matrix")
    if not np.allclose(matrix.sum(axis=0), 1.0, atol=1e-8):
        raise ValueError("A must be column-normalised")
    return -(matrix * log_stable(matrix)).sum(axis=0)


def kl_divergence(qo_u: np.ndarray, C: np.ndarray) -> float:
    """Return ``KL(qo_u || C)`` for two normalised distributions."""
    qo = _probability_vector(qo_u, name="qo_u")
    preference = _probability_vector(C, name="C", expected_size=qo.size)
    return float((log_stable(qo) - log_stable(preference)).dot(qo))


def infer_states(observation_index: int, A: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Infer ``q(s)`` from a discrete observation and a state prior."""
    matrix = np.asarray(A, dtype=float)
    state_prior = _probability_vector(prior, name="prior")
    if matrix.ndim != 2 or matrix.shape[1] != state_prior.size:
        raise ValueError("A must have shape (n_observations, n_states)")
    observation = _integer_index(observation_index, name="observation_index")
    if not 0 <= observation < matrix.shape[0]:
        raise IndexError(f"observation_index {observation_index} is out of range")
    if (matrix < 0).any() or not np.isfinite(matrix).all():
        raise ValueError("A must be finite and non-negative")
    if not np.allclose(matrix.sum(axis=0), 1.0, atol=1e-8):
        raise ValueError("A must be column-normalised")
    return softmax(log_stable(matrix[observation, :]) + log_stable(state_prior))


def _validate_model(A, B, C, qs_current):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    qs = _probability_vector(qs_current, name="qs_current")
    if A.ndim != 2 or A.shape[1] != qs.size:
        raise ValueError("A must have shape (n_observations, n_states)")
    if B.ndim != 3 or B.shape[:2] != (qs.size, qs.size):
        raise ValueError("B must have shape (n_states, n_states, n_actions)")
    if not np.isfinite(A).all() or (A < 0).any() or not np.allclose(A.sum(axis=0), 1.0, atol=1e-8):
        raise ValueError("A must be finite, non-negative, and column-stochastic")
    if not np.isfinite(B).all() or (B < 0).any() or not np.allclose(B.sum(axis=0), 1.0, atol=1e-8):
        raise ValueError("B must be finite, non-negative, and column-stochastic")
    if C is None:
        raise ValueError("C must be configured before planning")
    _probability_vector(C, name="C", expected_size=A.shape[0])
    return A, B, np.asarray(C, dtype=float), qs


def calculate_G(A, B, C, qs_current, actions) -> np.ndarray:
    """Return one-step expected free energy for every action."""
    A, B, C, qs = _validate_model(A, B, C, qs_current)
    if len(actions) != B.shape[2]:
        raise ValueError("actions length must equal B's action dimension")
    ambiguity = entropy(A)
    result = np.zeros(len(actions), dtype=float)
    for action_id in range(len(actions)):
        qs_u = get_expected_states(B, qs, action_id)
        qo_u = get_expected_observations(A, qs_u)
        result[action_id] = ambiguity.dot(qs_u) + kl_divergence(qo_u, C)
    return result


def _calculate_policy_terms(A, B, C, qs_current, policies):
    A, B, C, qs = _validate_model(A, B, C, qs_current)
    if not policies:
        raise ValueError("policies must be non-empty")
    ambiguity = entropy(A)
    epistemic = np.zeros(len(policies), dtype=float)
    pragmatic = np.zeros(len(policies), dtype=float)
    for policy_id, policy in enumerate(policies):
        policy = np.asarray(policy)
        if policy.ndim != 2 or policy.shape[1] != 1 or policy.shape[0] < 1:
            raise ValueError("each policy must have shape (policy_len, 1)")
        qs_pi = qs
        for timestep, action in enumerate(policy[:, 0]):
            qs_pi = get_expected_states(B, qs if timestep == 0 else qs_pi, action)
            qo_pi = get_expected_observations(A, qs_pi)
            epistemic[policy_id] += ambiguity.dot(qs_pi)
            pragmatic[policy_id] += kl_divergence(qo_pi, C)
    return epistemic, pragmatic


def calculate_G_policies(A, B, C, qs_current, policies) -> np.ndarray:
    """Return expected free energy summed over each policy horizon."""
    epistemic, pragmatic = _calculate_policy_terms(A, B, C, qs_current, policies)
    return epistemic + pragmatic


def calculate_G_policies_traced(
    A, B, C, qs_current, policies
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return policy EFE and its epistemic/pragmatic decomposition."""
    epistemic, pragmatic = _calculate_policy_terms(A, B, C, qs_current, policies)
    return epistemic + pragmatic, {"epistemic": epistemic, "pragmatic": pragmatic}


def compute_prob_actions(actions, policies, Q_pi: np.ndarray) -> np.ndarray:
    """Marginalise a policy posterior to the first action."""
    posterior = _probability_vector(Q_pi, name="Q_pi", expected_size=len(policies))
    if len(actions) == 0:
        raise ValueError("actions must be non-empty")
    probabilities = np.zeros(len(actions), dtype=float)
    for policy_id, policy in enumerate(policies):
        policy = np.asarray(policy)
        if policy.ndim != 2 or policy.shape[1] < 1 or policy.shape[0] < 1:
            raise ValueError("each policy must contain at least one action")
        action = _integer_index(policy[0, 0], name="policy action")
        if not 0 <= action < len(actions):
            raise IndexError(f"policy action {action} is out of range")
        probabilities[action] += posterior[policy_id]
    return norm_dist(probabilities)
