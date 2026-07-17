"""Tests for explicit CLI commands."""

import pytest

from blockference.cli import main


def test_cli_rejects_positional_run_arguments():
    with pytest.raises(SystemExit) as error:
        main(["3", "1", "1"])
    assert error.value.code == 2


def test_cli_pipeline_returns_success(tmp_path):
    code = main(["pipeline", "--config", "configs/smoke.yml", "--output-root", str(tmp_path)])
    assert code == 0
