"""Deterministic two-dimensional grid-world dynamics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from blockference.actions import DEFAULT_AFFORDANCES, validate_affordances
from blockference.gridference import _move


def _encode_identity(value: int | str) -> dict:
    if type(value) not in (int, str):
        raise TypeError("GridWorld agent identities must be built-in int or str")
    return {"type": type(value).__name__, "value": value}


def _decode_identity(payload: object) -> int | str:
    if not isinstance(payload, dict) or set(payload) != {"type", "value"}:
        raise ValueError("agent identity must contain type and value")
    value = payload["value"]
    if payload["type"] not in ("int", "str") or type(value).__name__ != payload["type"]:
        raise ValueError("agent identity tag does not match its int/str value")
    _encode_identity(value)
    return value


def _coordinate(value: Sequence[int], field_name: str, dimension: int) -> tuple[int, int]:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{field_name} must be a (y, x) coordinate")
    if any(isinstance(component, (bool, np.bool_)) or not isinstance(component, (int, np.integer)) for component in value):
        raise TypeError(f"{field_name} must contain integer coordinates")
    coordinate = (int(value[0]), int(value[1]))
    if not all(0 <= component < dimension for component in coordinate):
        raise ValueError(f"{field_name} {coordinate} is outside a {dimension}x{dimension} grid")
    return coordinate


def _agent_order(agent_ids: Sequence[int | str]) -> list[int | str]:
    """Preserve existing type-name/repr conflict priority, not registry order."""

    return sorted(agent_ids, key=lambda value: (type(value).__name__, repr(value)))


def resolve_moves(
    positions: Mapping[int | str, Sequence[int]],
    actions: Mapping[int | str, int],
    dimension: int,
    affordances: Sequence[str] = DEFAULT_AFFORDANCES,
) -> dict[int | str, tuple[int, int]]:
    """Resolve simultaneous actions without order-dependent collisions.

    A target occupied at the beginning of a step remains unavailable, so
    swaps and moves through another agent are rejected. If multiple agents
    propose the same free target, the deterministic identifier order grants
    it to the first proposal and leaves the others in place.
    """

    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 2:
        raise ValueError("dimension must be an integer >= 2")
    action_labels = validate_affordances(affordances)
    for agent_id in (*positions, *actions):
        _encode_identity(agent_id)
    if set(actions) != set(positions):
        raise ValueError("actions must contain exactly one action for every agent")
    current = {
        agent_id: _coordinate(position, f"positions[{agent_id!r}]", dimension)
        for agent_id, position in positions.items()
    }
    if len(set(current.values())) != len(current):
        raise ValueError("positions must contain a unique coordinate for every agent")
    proposed = {
        agent_id: _move(action, current[agent_id], dimension - 1, action_labels)
        for agent_id, action in actions.items()
    }
    occupied = set(current.values())
    result = dict(current)
    claimed: set[tuple[int, int]] = set()
    for agent_id in _agent_order(list(current)):
        source = current[agent_id]
        target = proposed[agent_id]
        if target != source and target in occupied:
            continue
        if target in claimed:
            continue
        result[agent_id] = target
        claimed.add(target)
    return result


class GridWorld:
    """Mutable grid with ordered built-in int/str identities and shared occupancy."""

    def __init__(
        self,
        dimension: int,
        positions: Mapping[int | str, Sequence[int]] | None = None,
        *,
        affordances: Sequence[str] = DEFAULT_AFFORDANCES,
        stochastic: bool = False,
    ) -> None:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 2:
            raise ValueError("dimension must be an integer >= 2")
        self.dimension = dimension
        self.border = dimension - 1
        self.affordances = validate_affordances(affordances)
        self.stochastic = bool(stochastic)
        self.positions: dict[int | str, tuple[int, int]] = {}
        for agent_id, position in (positions or {}).items():
            _encode_identity(agent_id)
            self.positions[agent_id] = _coordinate(position, f"positions[{agent_id!r}]", dimension)
        if len(set(self.positions.values())) != len(self.positions):
            raise ValueError("positions must contain a unique coordinate for every agent")

    @property
    def grid(self) -> tuple[tuple[int, int], ...]:
        """Return all valid coordinates in row-major ``(y, x)`` order."""

        return tuple((y, x) for y in range(self.dimension) for x in range(self.dimension))

    @property
    def current_state(self) -> dict[int | str, tuple[int, int]]:
        """Return a defensive copy of current agent positions."""

        return dict(self.positions)

    def step(self, actions: Mapping[int | str, int] | Sequence[int]) -> dict[int | str, tuple[int, int]]:
        """Apply one action per agent and return the updated positions."""

        if isinstance(actions, Mapping):
            action_map = dict(actions)
        else:
            if isinstance(actions, (str, bytes)) or len(actions) != len(self.positions):
                raise ValueError("a sequence action input must match the number of agents")
            action_map = dict(zip(self.positions, actions, strict=True))
        self.positions = resolve_moves(
            self.positions,
            action_map,
            self.dimension,
            self.affordances,
        )
        return self.current_state

    def serialize(self) -> dict:
        """Encode typed identities in registry order; reject unsupported Hashables."""
        return {
            "version": 2,
            "dimension": self.dimension,
            "affordances": list(self.affordances),
            "stochastic": self.stochastic,
            "positions": [
                {"agent_id": _encode_identity(agent_id), "position": list(position)}
                for agent_id, position in self.positions.items()
            ],
        }

    @classmethod
    def load(cls, payload: dict) -> GridWorld:
        """Reconstruct a :class:`GridWorld` from a :meth:`serialize` payload."""
        if not isinstance(payload, dict):
            raise TypeError("GridWorld payload must be a mapping")
        records = payload.get("positions")
        if payload.get("version") != 2 or not isinstance(records, list):
            raise ValueError("GridWorld requires version-2 ordered typed position records")
        positions = {}
        for record in records:
            if not isinstance(record, dict) or set(record) != {"agent_id", "position"}:
                raise ValueError("invalid GridWorld position record")
            agent_id = _decode_identity(record["agent_id"])
            if agent_id in positions:
                raise ValueError("duplicate GridWorld agent identity")
            positions[agent_id] = record["position"]
        return cls(
            payload["dimension"],
            positions,
            affordances=payload.get("affordances", DEFAULT_AFFORDANCES),
            stochastic=bool(payload.get("stochastic", False)),
        )


__all__ = ["GridWorld", "resolve_moves"]
