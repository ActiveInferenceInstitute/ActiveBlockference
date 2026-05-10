"""Smoke tests for blockference.simulations.

Real cadCAD/radCAD execution. Kept tiny (1 agent, 2 timesteps) so the
suite finishes in a couple of seconds.
"""

import pandas as pd
import pytest


def test_grid_sim_imports_silently():
    pytest.importorskip("radcad")
    import blockference.simulations.grid_sim  # noqa: F401


def test_run_grid_smoke(tmp_path):
    pytest.importorskip("radcad")
    from blockference.simulations import run_grid

    out = tmp_path / "out.csv"
    df = run_grid(
        dimension=3,
        no_agents=1,
        no_timesteps=2,
        output_path=str(out),
        seed=0,
    )
    assert isinstance(df, pd.DataFrame)
    assert out.exists()
    # cadCAD writes initial-state row + per-timestep rows
    assert len(df) >= 2
    for col in ("agents", "priors", "env_states", "actions", "inferences"):
        assert col in df.columns


def test_run_experiment_with_fixed_target(tmp_path):
    pytest.importorskip("radcad")
    from blockference.config import ExperimentConfig
    from blockference.simulations import run_experiment

    cfg = ExperimentConfig.from_dict({
        "name": "fixed_target",
        "seed": 1,
        "grid": {"dimension": 3, "planning_length": 2},
        "simulation": {
            "timesteps": 6,
            "n_agents": 1,
            "target": [2, 2],
            "initial_state": [0, 0],
        },
        "output": {"path": str(tmp_path / "fixed.csv")},
    })
    df = run_experiment(cfg)
    # Last row's env_states[0] should be the goal (or very close to it).
    last_env = df.iloc[-1]["env_states"]
    assert last_env[0] == (2, 2), f"agent did not reach target: ended at {last_env[0]}"


def test_run_experiment_writes_to_nested_path(tmp_path):
    pytest.importorskip("radcad")
    from blockference.config import ExperimentConfig
    from blockference.simulations import run_experiment

    out = tmp_path / "deep" / "nested" / "x.csv"
    cfg = ExperimentConfig.from_dict({
        "grid": {"dimension": 3},
        "simulation": {"timesteps": 1, "n_agents": 1, "target": [0, 1]},
        "output": {"path": str(out)},
    })
    run_experiment(cfg)
    assert out.exists()


def test_run_experiment_no_output(tmp_path, monkeypatch):
    pytest.importorskip("radcad")
    from blockference.config import ExperimentConfig
    from blockference.simulations import run_experiment

    monkeypatch.chdir(tmp_path)
    cfg = ExperimentConfig.from_dict({
        "grid": {"dimension": 3},
        "simulation": {"timesteps": 1, "n_agents": 1, "target": [0, 0]},
        "output": {"path": None},
    })
    df = run_experiment(cfg)
    assert isinstance(df, pd.DataFrame)
    # No CSV files written
    assert list(tmp_path.glob("*.csv")) == []
