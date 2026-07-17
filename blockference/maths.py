"""Validated NumPy primitives used by the Active Inference loop."""

from __future__ import annotations

from collections.abc import Sequence
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

MIN_PROB: float = 1e-16


def _as_finite_array(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain no non-finite values")
    return array


def _as_probability_array(value: np.ndarray, *, name: str) -> np.ndarray:
    array = _as_finite_array(value, name=name)
    if (array < 0).any():
        raise ValueError(f"{name} must not contain negative values")
    return array


def log_stable(x: np.ndarray) -> np.ndarray:
    """Return a finite natural logarithm for a non-negative array."""
    values = _as_probability_array(x, name="x")
    return np.log(np.clip(values, MIN_PROB, None))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return a numerically stable softmax along ``axis``."""
    values = _as_finite_array(x, name="x")
    if values.ndim == 0:
        return np.ones_like(values)
    if not -values.ndim <= axis < values.ndim:
        raise ValueError(f"axis {axis} is out of bounds for an array with {values.ndim} dimensions")
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    totals = exponentials.sum(axis=axis, keepdims=True)
    if (totals <= 0).any() or not np.isfinite(totals).all():
        raise ValueError("softmax could not construct a finite normalised distribution")
    return exponentials / totals


def norm_dist(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Normalise non-negative values along ``axis``.

    Zero-sum slices are represented by a uniform distribution.
    """
    values = _as_probability_array(x, name="x")
    if values.ndim == 0:
        raise ValueError("norm_dist requires an array with at least one dimension")
    if not -values.ndim <= axis < values.ndim:
        raise ValueError(f"axis {axis} is out of bounds for an array with {values.ndim} dimensions")
    total = values.sum(axis=axis, keepdims=True)
    size = values.shape[axis]
    if size == 0:
        raise ValueError("norm_dist cannot normalise an empty axis")
    uniform = np.full_like(values, 1.0 / size)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalised = np.divide(values, total, out=np.zeros_like(values), where=total > 0)
    return np.where(total == 0, uniform, normalised)


def sample(p: np.ndarray, rng: np.random.Generator | None = None) -> int:
    """Sample one index from a non-empty one-dimensional distribution."""
    probabilities = _as_probability_array(p, name="p")
    if probabilities.ndim != 1:
        raise ValueError(f"sample expects a one-dimensional vector, got {probabilities.ndim} dimensions")
    if probabilities.size == 0:
        raise ValueError("sample requires at least one probability")
    total = float(probabilities.sum())
    if total <= 0:
        raise ValueError("sample requires a positive probability total")
    probabilities = probabilities / total
    if rng is None:
        return int(np.random.choice(probabilities.size, p=probabilities))
    return int(rng.choice(probabilities.size, p=probabilities))


def onehot(index: int, size: int) -> np.ndarray:
    """Return a length-``size`` one-hot vector."""
    if not isinstance(index, (int, np.integer)):
        raise TypeError(f"onehot index must be an integer, got {type(index).__name__}")
    if not isinstance(size, (int, np.integer)) or size < 1:
        raise ValueError(f"onehot size must be a positive integer, got {size!r}")
    if not 0 <= int(index) < int(size):
        raise IndexError(f"onehot index {index} out of range [0, {size})")
    result = np.zeros(int(size), dtype=float)
    result[int(index)] = 1.0
    return result


def construct_policies(
    num_states: Sequence[int],
    num_controls: Sequence[int] | None = None,
    policy_len: int = 1,
    control_fac_idx: Sequence[int] | None = None,
) -> list[np.ndarray]:
    """Enumerate every policy for the supplied control-factor cardinalities."""
    if num_controls is None:
        raise ValueError("construct_policies: num_controls is required")
    if not isinstance(policy_len, (int, np.integer)) or policy_len < 1:
        raise ValueError(f"construct_policies: policy_len must be >= 1, got {policy_len!r}")
    if len(num_states) == 0 or len(num_controls) != len(num_states):
        raise ValueError("num_states and num_controls must be non-empty sequences of equal length")
    for count in num_states:
        if not isinstance(count, (int, np.integer)) or count < 1:
            raise ValueError("num_states entries must be positive integers")
    for count in num_controls:
        if not isinstance(count, (int, np.integer)) or count < 1:
            raise ValueError("num_controls entries must be positive integers")

    if control_fac_idx is None:
        controllable = set(range(len(num_states)))
    else:
        controllable = set(control_fac_idx)
        if any(
            not isinstance(index, (int, np.integer)) or not 0 <= int(index) < len(num_states)
            for index in controllable
        ):
            raise ValueError("control_fac_idx contains an invalid factor index")

    per_step_ranges = [
        range(int(num_controls[f])) if f in controllable else range(1)
        for f in range(len(num_states))
    ]
    per_step = list(product(*per_step_ranges))
    return [
        np.asarray(actions, dtype=int).reshape(int(policy_len), len(num_states))
        for actions in product(per_step, repeat=int(policy_len))
    ]
