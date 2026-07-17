"""Real backend and artefact round-trip tests."""

from __future__ import annotations

import pandas as pd

from blockference.config import ExperimentConfig
from blockference.io import validate_run_outputs
from blockference.pipeline import run_pipeline
from blockference.simulations.grid_sim import run_experiment


def config(engine: str = "radcad") -> ExperimentConfig:
    return ExperimentConfig.from_dict({
        "name": "test_run",
        "seed": 11,
        "engine": engine,
        "grid": {"dimension": 3, "planning_length": 1},
        "simulation": {"timesteps": 2, "runs": 1, "n_agents": 1, "target": [2, 2], "initial_state": [0, 0]},
        "output": {"path": None},
    })


def test_radcad_and_cadcad_emit_same_schema_and_seeded_states():
    radcad = run_experiment(config("radcad"))
    cadcad = run_experiment(config("cadcad"))
    assert list(radcad.columns) == list(cadcad.columns)
    assert radcad[["timestep", "substep", "run"]].equals(cadcad[["timestep", "substep", "run"]])
    repeated = run_experiment(config("radcad"))
    assert radcad["env_states"].astype(str).tolist() == repeated["env_states"].astype(str).tolist()


def test_pipeline_writes_and_validates_complete_tree(tmp_path):
    result = run_pipeline(config(), output_root=tmp_path, run_name="complete")
    assert result.ok
    assert result.validation_report.ok
    paths = result.paths
    required = [
        paths.config_path,
        paths.trajectory_csv,
        paths.summary_json,
        paths.matrices_json,
        paths.matrices_npz,
        paths.per_step_csv,
        paths.policies_json,
        paths.run_log,
        paths.validation_report,
        paths.viz_dir / "trajectory.png",
        paths.viz_dir / "action_distribution.png",
        paths.viz_dir / "efe.png",
        paths.animations_dir / "trajectory.gif",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in required)
    assert validate_run_outputs(paths).ok
    assert pd.read_csv(paths.trajectory_csv).index.tolist() == [0, 1, 2]


def test_pipeline_result_is_false_when_a_required_render_is_removed(tmp_path):
    result = run_pipeline(config(), output_root=tmp_path, run_name="rendered")
    result.paths.validation_report.unlink()
    result.paths.viz_dir.joinpath("efe.png").unlink()
    report = validate_run_outputs(result.paths)
    assert not report.ok
