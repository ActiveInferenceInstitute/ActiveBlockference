"""Real backend and artefact round-trip tests."""

from __future__ import annotations

import json

import pandas as pd

from blockference.config import ExperimentConfig
from blockference.io import (
    build_run_paths,
    parse_trajectory_records,
    persist_config,
    validate_run_outputs,
)
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
    values = {
        "simulation": {
            "timesteps": 2,
            "runs": 2,
            "n_agents": 2,
            "target": [2, 2],
            "initial_states": [[0, 0], [2, 2]],
        }
    }
    radcad_config = ExperimentConfig.from_dict({**config("radcad").to_dict(), **values})
    cadcad_config = ExperimentConfig.from_dict({**config("cadcad").to_dict(), **values})
    radcad = run_experiment(radcad_config)
    cadcad = run_experiment(cadcad_config)
    assert list(radcad.columns) == list(cadcad.columns)
    assert radcad[["timestep", "substep", "run"]].equals(cadcad[["timestep", "substep", "run"]])
    for column in (
        "env_states",
        "actions",
        "priors",
        "inferences",
        "efe",
        "efe_epistemic",
        "efe_pragmatic",
        "q_pi",
        "p_u",
        "obs_idx",
    ):
        assert radcad[column].astype(str).tolist() == cadcad[column].astype(str).tolist()
    # Typed schema parity: the two backends agree on normalized typed values,
    # not just stringified cells, and both round-trip deterministically.
    assert parse_trajectory_records(radcad) == parse_trajectory_records(cadcad)
    assert parse_trajectory_records(radcad) == parse_trajectory_records(run_experiment(radcad_config))
    first_run_config = ExperimentConfig.from_dict(
        {**radcad_config.to_dict(), "simulation": {**radcad_config.to_dict()["simulation"], "runs": 1}}
    )
    first_run = run_experiment(first_run_config)
    assert radcad.loc[radcad["run"] == 1, "env_states"].astype(str).tolist() == first_run[
        "env_states"
    ].astype(str).tolist()


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
        paths.manifest_json,
        paths.viz_dir / "trajectory.png",
        paths.viz_dir / "action_distribution.png",
        paths.viz_dir / "efe.png",
        paths.animations_dir / "trajectory.gif",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in required)
    assert validate_run_outputs(paths).ok
    original_summary = paths.summary_json.read_text(encoding="utf-8")
    paths.summary_json.write_text(original_summary + "tampered", encoding="utf-8")
    assert not validate_run_outputs(paths).ok
    paths.summary_json.write_text(original_summary, encoding="utf-8")
    paths.data_dir.joinpath("unexpected.bin").write_bytes(b"stale")
    assert not validate_run_outputs(paths).ok
    assert pd.read_csv(paths.trajectory_csv).index.tolist() == [0, 1, 2]


def test_pipeline_result_is_false_when_a_required_render_is_removed(tmp_path):
    result = run_pipeline(config(), output_root=tmp_path, run_name="rendered")
    result.paths.validation_report.unlink()
    result.paths.viz_dir.joinpath("efe.png").unlink()
    report = validate_run_outputs(result.paths)
    assert not report.ok


def test_pipeline_refuses_non_empty_run_tree_without_explicit_reuse(tmp_path):
    run_pipeline(config(), output_root=tmp_path, run_name="reused")
    try:
        run_pipeline(config(), output_root=tmp_path, run_name="reused")
    except FileExistsError:
        pass
    else:
        raise AssertionError("completed run tree was silently reused")


def test_interrupted_partial_tree_fails_closed(tmp_path):
    """A run interrupted before finishing can never yield a passing verdict."""
    paths = build_run_paths(tmp_path, "interrupted").ensure()
    persist_config(ExperimentConfig.from_dict({"output": {"path": None}}), paths)
    # Stale manifest claims completion but no data artefacts exist.
    paths.manifest_json.write_text(
        json.dumps({"version": 1, "complete": True, "files": []}), encoding="utf-8"
    )
    report = validate_run_outputs(paths)
    assert not report.ok
    assert any("present" in issue or "complete" in issue for issue in report.issues)


def test_leftover_atomic_temp_file_fails_closed(tmp_path):
    """A stale atomic-write temp file is treated as drift, never a valid artefact."""
    result = run_pipeline(config(), output_root=tmp_path, run_name="temp")
    result.paths.data_dir.joinpath(".trajectory.csv.deadbeef.tmp").write_bytes(b"partial")
    assert not validate_run_outputs(result.paths).ok


def test_validate_stages_are_focused_and_aggregate(tmp_path):
    from blockference.io.validate import (
        validate_artifact_presence,
        validate_config_boundary,
        validate_config_model_agreement,
        validate_config_trajectory_agreement,
        validate_manifest_boundary,
        validate_model_boundary,
        validate_per_step_boundary,
        validate_policies_boundary,
        validate_trajectory_boundary,
        validate_tree_structure,
    )

    result = run_pipeline(config(), output_root=tmp_path, run_name="stages")
    paths = result.paths
    assert validate_tree_structure(paths).ok
    assert validate_artifact_presence(paths).ok
    config_report, cfg = validate_config_boundary(paths)
    assert config_report.ok and cfg is not None
    trajectory_report, trajectory, summary = validate_trajectory_boundary(paths, cfg)
    assert trajectory_report.ok and trajectory is not None
    model_report, model_payload = validate_model_boundary(paths, cfg)
    assert model_report.ok and model_payload is not None
    assert validate_manifest_boundary(paths).ok
    assert validate_policies_boundary(paths).ok
    assert validate_per_step_boundary(paths, cfg, trajectory).ok
    assert validate_config_trajectory_agreement(cfg, trajectory, summary).ok
    assert validate_config_model_agreement(cfg, model_payload).ok
    assert validate_run_outputs(paths).ok
