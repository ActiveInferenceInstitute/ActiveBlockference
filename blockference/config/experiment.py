"""Typed configuration objects for ActiveBlockference experiments.

YAML or TOML files map onto :class:`ExperimentConfig`. The schema is kept
intentionally narrow so that adding a new key always requires a
deliberate edit to this file (and a matching test).

Example YAML
------------

.. code-block:: yaml

    name: gridworld_smoke
    seed: 42
    grid:
      dimension: 3
      planning_length: 2
      affordances: [UP, DOWN, LEFT, RIGHT, STAY]
    simulation:
      timesteps: 10
      runs: 1
      n_agents: 2
      target: random        # or [y, x]
      initial_state: [0, 0]
    output:
      path: results/gridworld_smoke.csv

"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

try:  # Python 3.11+
    import tomllib
except ImportError:  # pragma: no cover - py<3.11 fallback
    import tomli as tomllib  # type: ignore[no-redef]


__all__ = [
    "ExperimentConfig",
    "GridConfig",
    "SimulationConfig",
    "load_experiment_config",
]


@dataclass
class GridConfig:
    """Grid-world generative-model parameters."""

    dimension: int = 3
    planning_length: int = 2
    affordances: list[str] = field(default_factory=lambda: ["UP", "DOWN", "LEFT", "RIGHT", "STAY"])

    def __post_init__(self) -> None:
        if self.dimension < 2:
            raise ValueError(f"grid.dimension must be >= 2, got {self.dimension}")
        if self.planning_length < 1:
            raise ValueError(f"grid.planning_length must be >= 1, got {self.planning_length}")


@dataclass
class SimulationConfig:
    """cadCAD simulation parameters."""

    timesteps: int = 10
    runs: int = 1
    n_agents: int = 1
    target: str | tuple[int, int] = "random"
    initial_state: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        if self.timesteps < 1:
            raise ValueError(f"simulation.timesteps must be >= 1, got {self.timesteps}")
        if self.runs < 1:
            raise ValueError(f"simulation.runs must be >= 1, got {self.runs}")
        if self.n_agents < 1:
            raise ValueError(f"simulation.n_agents must be >= 1, got {self.n_agents}")
        # Coerce list-from-yaml to tuple
        if isinstance(self.initial_state, list):
            self.initial_state = tuple(self.initial_state)  # type: ignore[assignment]
        if isinstance(self.target, list):
            self.target = tuple(self.target)  # type: ignore[assignment]


@dataclass
class OutputConfig:
    """Where to persist artefacts."""

    path: str | None = "result.csv"


SUPPORTED_ENGINES: tuple[str, ...] = ("radcad", "cadcad")


@dataclass
class ExperimentConfig:
    """Top-level experiment description."""

    name: str = "unnamed"
    seed: int | None = None
    engine: str = "radcad"
    grid: GridConfig = field(default_factory=GridConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self) -> None:
        engine = (self.engine or "radcad").lower()
        if engine not in SUPPORTED_ENGINES:
            raise ValueError(f"unknown engine {self.engine!r}; expected one of {SUPPORTED_ENGINES}")
        self.engine = engine

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        grid_kwargs = data.get("grid", {}) or {}
        sim_kwargs = data.get("simulation", {}) or {}
        out_kwargs = data.get("output", {}) or {}
        return cls(
            name=data.get("name", "unnamed"),
            seed=data.get("seed"),
            engine=data.get("engine", "radcad"),
            grid=GridConfig(**grid_kwargs),
            simulation=SimulationConfig(**sim_kwargs),
            output=OutputConfig(**out_kwargs),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from a YAML or TOML file.

    The file format is detected from the suffix (``.yml``/``.yaml`` →
    YAML, ``.toml`` → TOML). Anything else raises ``ValueError``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".yml", ".yaml"}:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    elif suffix == ".toml":
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    else:
        raise ValueError(f"unsupported config extension {suffix!r}; use .yml, .yaml, or .toml")

    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping, got {type(data).__name__}")
    return ExperimentConfig.from_dict(data)
