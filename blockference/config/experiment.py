"""Strict, file-loadable experiment configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from blockference.actions import DEFAULT_AFFORDANCES, validate_affordances

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]


SUPPORTED_ENGINES: tuple[str, ...] = ("radcad", "cadcad")


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 1:
        raise ValueError(f"{field_name} must be >= 1, got {value}")
    return value


def _coordinate(value: Any, field_name: str) -> tuple[int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be a two-item coordinate")
    if len(value) != 2 or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ValueError(f"{field_name} must contain exactly two integer coordinates")
    return int(value[0]), int(value[1])


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return value


def _reject_unknown(data: Mapping[str, Any], allowed: set[str], field_name: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"unknown {field_name} key(s): {', '.join(unknown)}")


@dataclass
class GridConfig:
    """Grid-world generative-model parameters."""

    dimension: int = 3
    planning_length: int = 2
    max_policies: int = 100_000
    affordances: tuple[str, ...] = field(default_factory=lambda: DEFAULT_AFFORDANCES)

    def __post_init__(self) -> None:
        self.dimension = _positive_int(self.dimension, "grid.dimension")
        if self.dimension < 2:
            raise ValueError("grid.dimension must be >= 2")
        self.planning_length = _positive_int(self.planning_length, "grid.planning_length")
        self.max_policies = _positive_int(self.max_policies, "grid.max_policies")
        if isinstance(self.affordances, (str, bytes)):
            raise TypeError("grid.affordances must be a sequence of action labels")
        self.affordances = validate_affordances(self.affordances)


@dataclass
class SimulationConfig:
    """Simulation execution parameters."""

    timesteps: int = 10
    runs: int = 1
    n_agents: int = 1
    target: str | tuple[int, int] = "random"
    initial_state: tuple[int, int] = (0, 0)
    initial_states: tuple[tuple[int, int], ...] | None = None

    def __post_init__(self) -> None:
        self.timesteps = _positive_int(self.timesteps, "simulation.timesteps")
        self.runs = _positive_int(self.runs, "simulation.runs")
        self.n_agents = _positive_int(self.n_agents, "simulation.n_agents")
        self.initial_state = _coordinate(self.initial_state, "simulation.initial_state")
        if self.target != "random":
            self.target = _coordinate(self.target, "simulation.target")
        if self.initial_states is not None:
            if isinstance(self.initial_states, (str, bytes)):
                raise TypeError("simulation.initial_states must be a sequence of coordinates")
            try:
                values = tuple(
                    _coordinate(value, f"simulation.initial_states[{index}]")
                    for index, value in enumerate(self.initial_states)
                )
            except TypeError as exc:
                raise TypeError(
                    "simulation.initial_states must be a sequence of coordinates"
                ) from exc
            if len(values) != self.n_agents:
                raise ValueError(
                    "simulation.initial_states must contain exactly one coordinate per agent"
                )
            if len(set(values)) != len(values):
                raise ValueError("simulation.initial_states must contain unique coordinates")
            self.initial_states = values

    @property
    def resolved_initial_states(self) -> tuple[tuple[int, int], ...]:
        """Return one initial coordinate for every configured agent."""

        if self.initial_states is not None:
            return self.initial_states
        return (self.initial_state,) * self.n_agents


@dataclass
class OutputConfig:
    """Optional direct CSV output location for a simulation."""

    path: str | None = "result.csv"

    def __post_init__(self) -> None:
        if self.path is not None and not isinstance(self.path, str):
            raise TypeError("output.path must be a string or null")
        if self.path is not None and (not self.path.strip() or "\x00" in self.path):
            raise ValueError("output.path must be a non-empty path without NUL characters")


@dataclass
class ExperimentConfig:
    """Top-level experiment description with strict schema validation."""

    name: str = "unnamed"
    seed: int | None = None
    engine: str = "radcad"
    grid: GridConfig = field(default_factory=GridConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if self.name in {".", ".."} or any(char in self.name for char in "/\\\x00"):
            raise ValueError("name must be a single safe path component")
        if not all(char.isprintable() for char in self.name):
            raise ValueError("name must contain printable characters only")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise TypeError("seed must be an integer or null")
        if not isinstance(self.grid, GridConfig):
            raise TypeError("grid must be a GridConfig")
        if not isinstance(self.simulation, SimulationConfig):
            raise TypeError("simulation must be a SimulationConfig")
        if not isinstance(self.output, OutputConfig):
            raise TypeError("output must be an OutputConfig")
        if not isinstance(self.engine, str) or self.engine.lower() not in SUPPORTED_ENGINES:
            raise ValueError(f"unknown engine {self.engine!r}; expected one of {SUPPORTED_ENGINES}")
        self.engine = self.engine.lower()
        self._validate_coordinates()

    def _validate_coordinates(self) -> None:
        border = self.grid.dimension - 1
        for field_name, coordinate in (
            ("simulation.initial_state", self.simulation.initial_state),
        ):
            if not all(0 <= value <= border for value in coordinate):
                raise ValueError(f"{field_name} {coordinate} is outside the configured grid")
        for index, coordinate in enumerate(self.simulation.resolved_initial_states):
            if not all(0 <= value <= border for value in coordinate):
                raise ValueError(
                    f"simulation.initial_states[{index}] {coordinate} is outside the configured grid"
                )
        if self.simulation.target != "random":
            target = self.simulation.target
            assert isinstance(target, tuple)
            if not all(0 <= value <= border for value in target):
                raise ValueError(f"simulation.target {target} is outside the configured grid")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentConfig:
        """Build a configuration while rejecting every unknown key."""

        if not isinstance(data, Mapping):
            raise TypeError("config root must be a mapping")
        _reject_unknown(
            data, {"name", "seed", "engine", "grid", "simulation", "output"}, "configuration"
        )
        grid_data = _mapping(data.get("grid"), "grid")
        sim_data = _mapping(data.get("simulation"), "simulation")
        output_data = _mapping(data.get("output"), "output")
        _reject_unknown(
            grid_data,
            {"dimension", "planning_length", "max_policies", "affordances"},
            "grid",
        )
        _reject_unknown(
            sim_data,
            {"timesteps", "runs", "n_agents", "target", "initial_state", "initial_states"},
            "simulation",
        )
        _reject_unknown(output_data, {"path"}, "output")
        return cls(
            name=data.get("name", "unnamed"),
            seed=data.get("seed"),
            engine=data.get("engine", "radcad"),
            grid=GridConfig(**grid_data),
            simulation=SimulationConfig(**sim_data),
            output=OutputConfig(**output_data),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML/TOML-compatible nested mapping."""

        data = asdict(self)
        data["grid"]["affordances"] = list(data["grid"]["affordances"])
        data["simulation"]["initial_state"] = list(data["simulation"]["initial_state"])
        if data["simulation"]["initial_states"] is not None:
            data["simulation"]["initial_states"] = [
                list(coordinate) for coordinate in data["simulation"]["initial_states"]
            ]
        if data["simulation"]["target"] != "random":
            data["simulation"]["target"] = list(data["simulation"]["target"])
        return data


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a YAML or TOML experiment configuration."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    if config_path.suffix.lower() in {".yml", ".yaml"}:
        with config_path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
    elif config_path.suffix.lower() == ".toml":
        with config_path.open("rb") as stream:
            data = tomllib.load(stream)
    else:
        raise ValueError("unsupported config extension; use .yml, .yaml, or .toml")
    return ExperimentConfig.from_dict(data)


__all__ = [
    "SUPPORTED_ENGINES",
    "ExperimentConfig",
    "GridConfig",
    "OutputConfig",
    "SimulationConfig",
    "load_experiment_config",
]
