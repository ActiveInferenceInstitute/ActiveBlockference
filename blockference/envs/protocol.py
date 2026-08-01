"""Generic discrete-environment protocol for ActiveBlockference worlds.

The grid reference model is one concrete environment. Before another
environment is added, it must satisfy this contract so that simulation,
persistence, and validation can treat every environment uniformly instead of
special-casing the grid. The protocol deliberately covers:

* state identity (a ``positions`` mapping of agent → coordinate),
* observations (the ``grid`` coordinate set and ``affordances`` vocabulary),
* simultaneous actions and collision resolution (``step``),
* a frozen ``current_state`` snapshot,
* transition stochasticity (``stochastic`` flag),
* and lossless serialization (``serialize`` / ``load``).
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class DiscreteEnvironment(Protocol):
    """Structural contract every environment must satisfy.

    ``@runtime_checkable`` lets tooling assert ``isinstance(world,
    DiscreteEnvironment)`` while keeping the contract structural (no forced
    inheritance), so a new environment only needs to provide the members below.
    """

    dimension: int
    border: int
    affordances: Sequence[str]
    positions: Mapping[Hashable, tuple[int, int]]
    stochastic: bool

    @property
    def grid(self) -> Sequence[tuple[int, int]]: ...

    @property
    def current_state(self) -> Mapping[Hashable, tuple[int, int]]: ...

    def step(
        self, actions: Mapping[Hashable, int] | Sequence[int]
    ) -> Mapping[Hashable, tuple[int, int]]: ...

    def serialize(self) -> dict: ...

    @classmethod
    def load(cls, payload: dict) -> DiscreteEnvironment: ...


__all__ = ["DiscreteEnvironment"]
