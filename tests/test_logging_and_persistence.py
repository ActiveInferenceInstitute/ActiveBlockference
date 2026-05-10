"""Tests for logging_setup + new persistence/validation helpers."""

import json
import logging

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# logging_setup
# ---------------------------------------------------------------------------


def test_get_logger_returns_namespaced_logger():
    from blockference.logging_setup import get_logger

    log = get_logger("blockference.test")
    assert isinstance(log, logging.Logger)
    assert log.name == "blockference.test"


def test_configure_run_logging_writes_file(tmp_path):
    from blockference.logging_setup import (
        configure_run_logging,
        get_logger,
        remove_handler,
    )

    log_path = tmp_path / "subdir" / "run.log"
    handler = configure_run_logging(log_path)
    log = get_logger("blockference.test_persistence")
    log.info("hello world")
    handler.flush()
    assert log_path.exists()
    assert "hello world" in log_path.read_text()
    remove_handler(handler)


# ---------------------------------------------------------------------------
# persist_generative_model + persist_per_step_records + persist_policies
# ---------------------------------------------------------------------------


def test_persist_generative_model(tmp_path):
    from blockference.gridference import ActiveGridference, make_grid
    from blockference.io import build_run_paths, persist_generative_model

    grid = make_grid(3, 2)
    a = ActiveGridference(grid)
    a.get_C((2, 2))
    a.get_D((0, 0))

    paths = build_run_paths(root=tmp_path, run_name="r")
    out = persist_generative_model({0: a}, paths)
    assert out.exists()
    payload = json.loads(out.read_text())
    assert "0" in payload
    assert payload["0"]["n_states"] == 9
    assert len(payload["0"]["A"]) == 9
    assert len(payload["0"]["B"]) == 9
    assert payload["0"]["E"] == ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]


def test_persist_policies(tmp_path):
    from blockference.io import build_run_paths, persist_policies
    from blockference.maths import construct_policies

    paths = build_run_paths(root=tmp_path, run_name="r")
    ps = construct_policies([4], [3], policy_len=2)
    out = persist_policies(ps, paths)
    payload = json.loads(out.read_text())
    assert payload["n_policies"] == 9
    assert len(payload["policies"]) == 9


def test_persist_per_step_records(tmp_path):
    from blockference.io import build_run_paths, persist_per_step_records

    paths = build_run_paths(root=tmp_path, run_name="r")
    records = [
        {"agent_id": 0, "timestep": 0, "action": 1, "posterior_entropy": 0.5},
        {"agent_id": 0, "timestep": 1, "action": 2, "posterior_entropy": 0.4},
    ]
    out = persist_per_step_records(records, paths)
    df = pd.read_csv(out)
    assert len(df) == 2
    assert "posterior_entropy" in df.columns


# ---------------------------------------------------------------------------
# validate_generative_model
# ---------------------------------------------------------------------------


def test_validate_generative_model_passes_for_valid_agent():
    from blockference.gridference import ActiveGridference, make_grid
    from blockference.io import validate_generative_model

    grid = make_grid(3, 2)
    a = ActiveGridference(grid)
    a.get_C((2, 2))
    a.get_D((0, 0))
    rep = validate_generative_model(a)
    assert rep.ok, f"unexpected validation issues: {rep.issues}"
    assert rep.checks["A_column_stochastic"]
    assert rep.checks["B_column_stochastic_per_action"]
    assert rep.checks["C_normalised"]
    assert rep.checks["D_normalised"]


def test_validate_generative_model_flags_unnormalised_C(tmp_path):
    from blockference.gridference import ActiveGridference, make_grid
    from blockference.io import validate_generative_model

    grid = make_grid(3, 2)
    a = ActiveGridference(grid)
    a.get_C((2, 2))
    a.get_D((0, 0))
    a.C = a.C * 2.0  # break normalisation
    rep = validate_generative_model(a)
    assert not rep.ok
    assert any("C_normalised" in issue for issue in rep.issues)


# ---------------------------------------------------------------------------
# Pipeline emits all the new artefacts
# ---------------------------------------------------------------------------


def test_pipeline_writes_run_log_and_extras(tmp_path):
    pytest.importorskip("radcad")
    from blockference.config import ExperimentConfig
    from blockference.pipeline import run_pipeline

    cfg = ExperimentConfig.from_dict({
        "name": "logged",
        "seed": 0,
        "grid": {"dimension": 3, "planning_length": 1},
        "simulation": {
            "timesteps": 2,
            "n_agents": 1,
            "target": [2, 2],
            "initial_state": [0, 0],
        },
        "output": {"path": "ignored.csv"},
    })
    result = run_pipeline(cfg, output_root=tmp_path, run_name="r")
    p = result.paths
    assert p.run_log.exists() and p.run_log.stat().st_size > 0
    assert p.matrices_json.exists()
    assert p.per_step_csv.exists()
    assert p.policies_json.exists()
    # New report field is populated
    assert "0" in result.model_reports
    assert result.model_reports["0"].ok
