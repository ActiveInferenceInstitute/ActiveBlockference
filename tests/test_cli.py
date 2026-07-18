"""Tests for explicit CLI commands."""

import json

import pytest

from blockference.cli import main


def test_cli_rejects_positional_run_arguments():
    with pytest.raises(SystemExit) as error:
        main(["3", "1", "1"])
    assert error.value.code == 2


def test_cli_pipeline_returns_success(tmp_path, capsys):
    code = main(["pipeline", "--config", "configs/smoke.yml", "--output-root", str(tmp_path)])
    assert code == 0
    output = capsys.readouterr().out.strip().splitlines()[-1]
    assert json.loads(output)["ok"] is True


def test_cli_run_and_validation_emit_json(tmp_path, capsys):
    output = tmp_path / "run.csv"
    assert main(["run", "--config", "configs/smoke.yml", "--output", str(output)]) == 0
    run_payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert run_payload["ok"] is True
    pipeline_root = tmp_path / "pipeline"
    assert main(
        [
            "pipeline",
            "--config",
            "configs/smoke.yml",
            "--output-root",
            str(pipeline_root),
            "--run-name",
            "cli",
        ]
    ) == 0
    capsys.readouterr()
    assert main(["validation", "--run-dir", str(pipeline_root / "cli")]) == 0
    validation_payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert validation_payload["ok"] is True


def test_cadcad_cli_stdout_is_one_json_document(tmp_path, capsys):
    config = tmp_path / "cadcad.yml"
    config.write_text(
        "name: cadcad_cli\n"
        "seed: 3\n"
        "engine: cadcad\n"
        "grid:\n  dimension: 2\n  planning_length: 1\n"
        "simulation:\n  timesteps: 1\n  runs: 1\n  n_agents: 1\n  target: [1, 1]\n  initial_state: [0, 0]\n"
        "output:\n  path: null\n",
        encoding="utf-8",
    )
    assert main(["run", "--config", str(config), "--output", str(tmp_path / "cadcad.csv")]) == 0
    output = capsys.readouterr().out.strip().splitlines()
    assert len(output) == 1
    assert json.loads(output[0])["ok"] is True
