"""Tests for blockference.config."""

import pytest

from blockference.config import (
    ExperimentConfig,
    GridConfig,
    SimulationConfig,
    load_experiment_config,
)

# ---------------------------------------------------------------------------
# Dataclass validation
# ---------------------------------------------------------------------------


def test_grid_config_defaults():
    g = GridConfig()
    assert g.dimension == 3
    assert g.planning_length == 2
    assert g.affordances[0] == "UP"


def test_grid_config_validates_dimension():
    with pytest.raises(ValueError, match="dimension"):
        GridConfig(dimension=1)


def test_grid_config_validates_planning_length():
    with pytest.raises(ValueError, match="planning_length"):
        GridConfig(planning_length=0)


def test_simulation_config_validates_positive_ints():
    with pytest.raises(ValueError, match="timesteps"):
        SimulationConfig(timesteps=0)
    with pytest.raises(ValueError, match="runs"):
        SimulationConfig(runs=0)
    with pytest.raises(ValueError, match="n_agents"):
        SimulationConfig(n_agents=0)


def test_simulation_config_coerces_lists_to_tuples():
    s = SimulationConfig(initial_state=[1, 2], target=[3, 4])  # type: ignore[arg-type]
    assert s.initial_state == (1, 2)
    assert s.target == (3, 4)


def test_experiment_config_from_dict_round_trip():
    cfg = ExperimentConfig.from_dict(
        {
            "name": "t",
            "seed": 1,
            "grid": {"dimension": 4, "planning_length": 3},
            "simulation": {"timesteps": 5, "runs": 2, "n_agents": 3, "target": [2, 2]},
            "output": {"path": "out.csv"},
        }
    )
    d = cfg.to_dict()
    assert d["name"] == "t"
    assert d["grid"]["dimension"] == 4
    assert d["simulation"]["target"] == (2, 2)


def test_experiment_config_rejects_unknown_keys():
    with pytest.raises(TypeError):
        ExperimentConfig.from_dict({"grid": {"unknown_key": 99}})


# ---------------------------------------------------------------------------
# YAML / TOML loaders
# ---------------------------------------------------------------------------


def test_load_yaml_smoke_config():
    cfg = load_experiment_config("configs/smoke.yml")
    assert cfg.name == "smoke"
    assert cfg.simulation.target == (2, 2)
    assert cfg.simulation.initial_state == (0, 0)


def test_load_toml_sweep_config():
    cfg = load_experiment_config("configs/sweep.toml")
    assert cfg.name == "gridworld_sweep"
    assert cfg.simulation.runs == 3
    assert cfg.grid.dimension == 4


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_experiment_config(tmp_path / "nope.yml")


def test_load_unknown_extension_raises(tmp_path):
    bad = tmp_path / "x.json"
    bad.write_text("{}")
    with pytest.raises(ValueError, match="extension"):
        load_experiment_config(bad)


def test_load_non_mapping_raises(tmp_path):
    bad = tmp_path / "x.yml"
    bad.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError, match="mapping"):
        load_experiment_config(bad)


def test_yaml_and_toml_produce_same_shape(tmp_path):
    yml = tmp_path / "a.yml"
    yml.write_text("name: x\ngrid:\n  dimension: 3\nsimulation:\n  timesteps: 1\n  n_agents: 1\n")
    toml = tmp_path / "a.toml"
    toml.write_text(
        'name = "x"\n[grid]\ndimension = 3\n[simulation]\ntimesteps = 1\nn_agents = 1\n'
    )
    a = load_experiment_config(yml)
    b = load_experiment_config(toml)
    assert a.to_dict() == b.to_dict()
