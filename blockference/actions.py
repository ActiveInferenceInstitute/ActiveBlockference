"""Canonical action vocabulary for two-dimensional grid worlds."""

from __future__ import annotations

from collections.abc import Iterable

DEFAULT_AFFORDANCES: tuple[str, ...] = ("UP", "DOWN", "LEFT", "RIGHT", "STAY")
SUPPORTED_AFFORDANCES: frozenset[str] = frozenset(DEFAULT_AFFORDANCES)


def validate_affordances(affordances: Iterable[str] | None) -> tuple[str, ...]:
    """Return a validated, ordered action vocabulary."""

    values = tuple(DEFAULT_AFFORDANCES if affordances is None else affordances)
    if not values:
        raise ValueError("affordances must contain at least one action")
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("affordances must contain non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError("affordances must not contain duplicates")
    unsupported = set(values) - SUPPORTED_AFFORDANCES
    if unsupported:
        raise ValueError(
            f"unsupported affordances: {sorted(unsupported)}; "
            f"choose from {sorted(SUPPORTED_AFFORDANCES)}"
        )
    return values


__all__ = ["DEFAULT_AFFORDANCES", "SUPPORTED_AFFORDANCES", "validate_affordances"]
