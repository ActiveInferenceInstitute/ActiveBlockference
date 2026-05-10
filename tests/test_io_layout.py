"""Tests for blockference.io layout + persistence + validation."""

import json

import pandas as pd

from blockference.config import ExperimentConfig
from blockference.io import (
    build_run_paths,
    persist_config,
    persist_dataframe,
    persist_summary,
    validate_run_outputs,
    validate_trajectory_dataframe,
)


def test_build_run_paths_creates_full_tree(tmp_path):
    paths = build_run_paths(root=tmp_path, run_name="r")
    assert paths.run_dir.is_dir()
    assert paths.data_dir.is_dir()
    assert paths.viz_dir.is_dir()
    assert paths.animations_dir.is_dir()


def test_build_run_paths_timestamp_suffix(tmp_path):
    p = build_run_paths(root=tmp_path, run_name="t", timestamped=True)
    assert p.run_name.startswith("t-")
    assert "T" in p.run_name and p.run_name.endswith("Z")


def test_persist_config_writes_yaml(tmp_path):
    cfg = ExperimentConfig.from_dict({"name": "x", "grid": {"dimension": 3}})
    paths = build_run_paths(root=tmp_path, run_name="p")
    out = persist_config(cfg, paths)
    assert out.exists()
    assert "name: x" in out.read_text()


def test_persist_dataframe_csv_at_minimum(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    paths = build_run_paths(root=tmp_path, run_name="p")
    written = persist_dataframe(df, paths)
    assert "csv" in written
    reloaded = pd.read_csv(written["csv"])
    assert list(reloaded.columns) == ["a", "b"]


def test_persist_summary_round_trip(tmp_path):
    paths = build_run_paths(root=tmp_path, run_name="p")
    summary = {"n_agents": 2, "final_positions": {"0": (1, 2)}}
    out = persist_summary(summary, paths)
    loaded = json.loads(out.read_text())
    assert loaded["n_agents"] == 2


def test_validate_trajectory_dataframe_flags_empty():
    df = pd.DataFrame()
    report = validate_trajectory_dataframe(df)
    assert not report.ok


def test_validate_trajectory_dataframe_required_columns():
    df = pd.DataFrame({"agents": [{0: object()}], "priors": [{}]})
    report = validate_trajectory_dataframe(df)
    assert "required_columns" in report.checks
    assert not report.checks["required_columns"]


def test_validate_run_outputs_round_trip(tmp_path):
    cfg = ExperimentConfig.from_dict({"name": "x"})
    paths = build_run_paths(root=tmp_path, run_name="r")
    persist_config(cfg, paths)
    persist_dataframe(pd.DataFrame({"a": [1]}), paths)
    persist_summary({"k": 1}, paths)
    report = validate_run_outputs(paths)
    assert report.ok
    out = report.write(paths)
    assert out.exists()
