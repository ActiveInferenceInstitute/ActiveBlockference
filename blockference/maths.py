"""Self-contained Active Inference math primitives.

Why this module exists
----------------------

`inferactively-pymdp` made a major API break at v1.0 (a JAX/Equinox
rewrite). Symbols ActiveBlockference relied on — ``softmax``,
``spm_log_single``, ``sample``, ``onehot``, ``norm_dist``,
``construct_policies`` returning a NumPy array — either disappeared,
moved, or now return JAX arrays.

Rather than chase upstream forever, we own the small, well-defined set
of math primitives we actually need. Every function here:

* takes and returns ``numpy.ndarray`` (no JAX),
* is deterministic given a seeded RNG,
* is unit-tested in ``tests/test_maths.py``.

We still depend on pymdp 1.0.x — but only for high-level objects (the
``Agent`` class and its companions) that callers can use directly if
they want; ActiveBlockference's grid pipeline does not require it.
"""

from __future__ import annotations

from itertools import product

import numpy as np

__all__ = [
    "MIN_PROB",
    "construct_policies",
    "log_stable",
    "norm_dist",
    "onehot",
    "sample",
    "softmax",
]


#: Numerical floor used to keep ``log`` and probability ratios finite.
MIN_PROB: float = 1e-16


# ---------------------------------------------------------------------------
# Stable log / softmax / normalisation
# ---------------------------------------------------------------------------


def log_stable(x: np.ndarray) -> np.ndarray:
    """Natural log clipped at :data:`MIN_PROB` to avoid ``-inf``."""
    x = np.asarray(x, dtype=float)
    return np.log(np.clip(x, MIN_PROB, None))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically-stable softmax along ``axis``.

    Subtracts the per-axis max before exponentiating, which keeps the
    intermediate values in a representable range for any input magnitude.
    """
    x = np.asarray(x, dtype=float)
    x_shift = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x_shift)
    return e / np.sum(e, axis=axis, keepdims=True)


def norm_dist(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Normalise ``x`` along ``axis`` so each slice sums to 1.

    Zero-sum slices are left as uniform distributions (rather than
    propagating ``nan``s), matching the behaviour of pymdp's legacy
    ``norm_dist`` helper.
    """
    x = np.asarray(x, dtype=float)
    total = x.sum(axis=axis, keepdims=True)
    safe_total = np.where(total > 0, total, 1.0)
    out = x / safe_total
    if np.any(total == 0):
        # Fill the zero-sum slices with a uniform distribution.
        size = x.shape[axis]
        zero_mask = total == 0
        out = np.where(zero_mask, np.full_like(out, 1.0 / size), out)
    return out


# ---------------------------------------------------------------------------
# Sampling and one-hot construction
# ---------------------------------------------------------------------------


def sample(p: np.ndarray, rng: np.random.Generator | None = None) -> int:
    """Sample a single index from a 1-D categorical distribution ``p``.

    Parameters
    ----------
    p :
        1-D probability vector. Will be re-normalised if it does not sum
        to one within ``1e-9`` to mirror legacy pymdp behaviour.
    rng :
        Optional :class:`numpy.random.Generator`. When omitted, the call
        falls back to NumPy's *global* RNG so that callers seeding via
        ``np.random.seed(...)`` (the legacy pymdp idiom) still get
        reproducible draws.
    """
    p = np.asarray(p, dtype=float).ravel()
    if not np.isfinite(p).all():
        raise ValueError("sample: input contains non-finite probabilities")
    s = p.sum()
    if not np.isclose(s, 1.0, atol=1e-9):
        p = p / s if s > 0 else np.full_like(p, 1.0 / p.size)
    if rng is None:
        return int(np.random.choice(p.size, p=p))
    return int(rng.choice(p.size, p=p))


def onehot(index: int, size: int) -> np.ndarray:
    """Length-``size`` one-hot vector with a 1 at ``index``."""
    if not (0 <= index < size):
        raise IndexError(f"onehot index {index} out of range [0, {size})")
    v = np.zeros(size, dtype=float)
    v[index] = 1.0
    return v


# ---------------------------------------------------------------------------
# Policy construction
# ---------------------------------------------------------------------------


def construct_policies(
    num_states,
    num_controls=None,
    policy_len: int = 1,
    control_fac_idx=None,
) -> list[np.ndarray]:
    """Enumerate every length-``policy_len`` policy for the given factors.

    Drop-in numpy replacement for ``pymdp.control.construct_policies``.
    Each returned policy is a ``(policy_len, num_factors)`` int array
    where ``policy[t, f]`` is the action index for factor ``f`` at time
    ``t``.

    Parameters
    ----------
    num_states :
        Sequence ``[n_factors]``. Only its length is used.
    num_controls :
        Sequence ``[n_actions_per_factor]``. Required.
    policy_len :
        Planning horizon. Must be >= 1.
    control_fac_idx :
        Optional subset of factor indices that are controllable. If
        ``None``, all factors are assumed controllable.
    """
    if num_controls is None:
        raise ValueError("construct_policies: num_controls is required")
    if policy_len < 1:
        raise ValueError(f"construct_policies: policy_len must be >= 1, got {policy_len}")

    n_factors = len(num_states)
    if control_fac_idx is None:
        control_fac_idx = list(range(n_factors))

    # For each timestep, the action vector of length n_factors. Non-controllable
    # factors are forced to action 0.
    action_ranges_per_factor = []
    for f in range(n_factors):
        if f in control_fac_idx:
            action_ranges_per_factor.append(range(num_controls[f]))
        else:
            action_ranges_per_factor.append(range(1))

    per_step_combos = list(product(*action_ranges_per_factor))
    policies = []
    for combo in product(per_step_combos, repeat=policy_len):
        # combo is a tuple of length policy_len, each element a tuple of action indices
        policy = np.array(combo, dtype=int).reshape(policy_len, n_factors)
        policies.append(policy)
    return policies
