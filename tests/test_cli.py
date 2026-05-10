"""Tests for the grid_sim CLI entry point."""

import pytest


def test_cli_with_config_file(tmp_path):
    pytest.importorskip("radcad")
    from blockference.simulations.grid_sim import main

    cfg = tmp_path / "c.yml"
    cfg.write_text(
        "name: cli_test\n"
        "grid:\n"
        "  dimension: 3\n"
        "  planning_length: 1\n"
        "simulation:\n"
        "  timesteps: 1\n"
        "  n_agents: 1\n"
        "  target: [0, 1]\n"
        "output:\n"
        f"  path: {tmp_path / 'cli.csv'}\n"
    )
    rc = main([str(cfg)])
    assert rc == 0
    assert (tmp_path / "cli.csv").exists()


def test_cli_legacy_positional(tmp_path, monkeypatch):
    pytest.importorskip("radcad")
    from blockference.simulations.grid_sim import main

    monkeypatch.chdir(tmp_path)
    out = tmp_path / "legacy.csv"
    rc = main(["3", "1", "1", "-o", str(out)])
    assert rc == 0
    assert out.exists()


def test_cli_no_args_returns_help_code(capsys):
    from blockference.simulations.grid_sim import main

    rc = main([])
    assert rc == 2
