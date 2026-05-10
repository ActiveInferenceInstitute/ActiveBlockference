"""End-to-end tests for blockference.pipeline.run_pipeline."""

import json

import pytest


def test_run_pipeline_emits_full_tree(tmp_path):
    pytest.importorskip("radcad")
    pytest.importorskip("PIL")
    from blockference.config import ExperimentConfig
    from blockference.pipeline import run_pipeline

    cfg = ExperimentConfig.from_dict({
        "name": "pipe_test",
        "seed": 0,
        "grid": {"dimension": 3, "planning_length": 1},
        "simulation": {
            "timesteps": 3,
            "n_agents": 1,
            "target": [2, 2],
            "initial_state": [0, 0],
        },
        "output": {"path": "ignored.csv"},
    })
    result = run_pipeline(cfg, output_root=tmp_path, run_name="r")

    paths = result.paths
    assert paths.config_path.exists()
    assert paths.trajectory_csv.exists()
    assert paths.summary_json.exists()
    assert paths.validation_report.exists()
    # at least one viz + one animation produced
    assert (paths.viz_dir / "trajectory.png").exists()
    assert (paths.animations_dir / "trajectory.gif").exists()

    summary = json.loads(paths.summary_json.read_text())
    assert summary["n_agents"] == 1
    assert summary["grid_dimension"] == 3

    report = json.loads(paths.validation_report.read_text())
    assert report["ok"] is True

    assert result.schema_report.ok
    assert result.artefacts_report.ok


def test_run_pipeline_with_yaml_config_path(tmp_path):
    pytest.importorskip("radcad")
    from blockference.pipeline import run_pipeline

    cfg = tmp_path / "c.yml"
    cfg.write_text(
        "name: pipe_yaml\n"
        "seed: 1\n"
        "grid:\n  dimension: 3\n  planning_length: 1\n"
        "simulation:\n  timesteps: 2\n  n_agents: 1\n  target: [0, 1]\n  initial_state: [0, 0]\n"
        "output:\n  path: x.csv\n"
    )
    result = run_pipeline(cfg, output_root=tmp_path, run_name="from_yaml")
    assert result.paths.run_dir.is_dir()
    assert result.artefacts_report.ok
