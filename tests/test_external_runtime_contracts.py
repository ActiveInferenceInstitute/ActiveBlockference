"""Runtime-selection and invocation-output contracts, not GNN/package qualification."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import scripts.manuscript_variables as variables
import scripts.release_check as gate
from blockference.gnn_runtime import resolve_gnn_runtime


@pytest.fixture
def runtimes(tmp_path, monkeypatch):
    for name in ("GNN_ROOT", "GNN_RUNTIME_PYTHON", "GNN_PYTHON"):
        monkeypatch.delenv(name, raising=False)
    paths = {}
    for name in ("cli", "environment", "legacy"):
        path = tmp_path / f"{name}-python"
        path.write_text("selection fixture, not an executable runtime\n", encoding="utf-8")
        paths[name] = path
    checkout = tmp_path / "checkout"
    python = checkout / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python.parent.mkdir(parents=True)
    python.write_text("checkout interpreter selection fixture\n", encoding="utf-8")
    paths.update(checkout=checkout, checkout_python=python)
    return paths


def test_shared_selector_has_one_precedence_without_project_python_fallback(runtimes, monkeypatch):
    monkeypatch.setenv("GNN_ROOT", str(runtimes["checkout"]))
    monkeypatch.setenv("GNN_PYTHON", str(runtimes["legacy"]))
    monkeypatch.setenv("GNN_RUNTIME_PYTHON", str(runtimes["environment"]))
    assert resolve_gnn_runtime(runtimes["cli"], python=runtimes["legacy"]) == str(runtimes["cli"])
    assert resolve_gnn_runtime(python=runtimes["legacy"]) == str(runtimes["environment"])
    monkeypatch.delenv("GNN_RUNTIME_PYTHON")
    assert resolve_gnn_runtime(python=runtimes["cli"]) == str(runtimes["cli"])
    assert resolve_gnn_runtime() == str(runtimes["legacy"])
    monkeypatch.delenv("GNN_PYTHON")
    assert resolve_gnn_runtime() == str(runtimes["checkout_python"])
    monkeypatch.delenv("GNN_ROOT")
    assert resolve_gnn_runtime(required=False) is None
    with pytest.raises(FileNotFoundError):
        resolve_gnn_runtime()


@pytest.mark.parametrize("explicit", ["missing", "empty"])
def test_invalid_explicit_runtime_never_falls_back_to_an_available_environment(runtimes, monkeypatch, explicit):
    monkeypatch.setenv("GNN_RUNTIME_PYTHON", str(runtimes["environment"]))
    selection = str(runtimes["cli"].parent / "missing") if explicit == "missing" else ""
    with pytest.raises(FileNotFoundError):
        resolve_gnn_runtime(selection)


def test_venv_launcher_symlink_is_not_replaced_by_its_base_interpreter(runtimes):
    launcher = runtimes["cli"].parent / "venv-python"
    launcher.symlink_to(runtimes["cli"])
    assert resolve_gnn_runtime(launcher) == str(launcher)
    assert resolve_gnn_runtime(launcher) != str(launcher.resolve())


def test_gate_records_missing_external_runtime_before_executing_any_arm(tmp_path, monkeypatch, capsys):
    for name in ("GNN_ROOT", "GNN_RUNTIME_PYTHON", "GNN_PYTHON"):
        monkeypatch.delenv(name, raising=False)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a gate command executed before required runtime admission")

    monkeypatch.setattr(gate, "_run", forbidden)
    monkeypatch.setattr(gate, "_changed_paths", forbidden)
    receipt = tmp_path / "receipt.json"
    assert gate.main(["--arm", "manuscript", "--timings", str(receipt), "--json"]) == 1
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["arms"]["manuscript"]["status"] == "failed"
    summary = next(line for line in capsys.readouterr().out.splitlines() if line.startswith('{"arms":'))
    assert json.loads(summary)["ok"] is False


def _manuscript_sources(root):
    (root / "pyproject.toml").write_text('[project]\nversion="1.0.0"\nrequires-python=">=3.11"\n', encoding="utf-8")
    workflow = root / ".github" / "workflows" / "ci.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("jobs:\n  gnn-conformance:\n    name: GNN conformance\n", encoding="utf-8")
    selected = root / "configs" / "alternate.toml"
    selected.parent.mkdir()
    selected.write_text('engine="CaDcAd"\n[grid]\ndimension=4\nplanning_length=1\n', encoding="utf-8")
    metadata = root / "docs" / "manuscript" / "config.yaml"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(yaml.safe_dump({"experiment": {"config_path": "configs/alternate.toml"}}), encoding="utf-8")
    return metadata, workflow


@pytest.mark.parametrize("selection", ["cli", "checkout"])
def test_manuscript_variables_consume_the_selected_runtime_and_typed_config(
    tmp_path, runtimes, monkeypatch, selection,
):
    root = tmp_path / "publication"
    root.mkdir()
    metadata, workflow = _manuscript_sources(root)
    monkeypatch.setattr(variables, "ROOT", root)
    monkeypatch.setattr(variables, "MANUSCRIPT_CONFIG_PATH", metadata)
    monkeypatch.setattr(variables, "WORKFLOW_PATH", workflow)
    if selection == "cli":
        monkeypatch.setenv("GNN_RUNTIME_PYTHON", str(runtimes["environment"]))
        expected = str(runtimes["cli"])
        argument = expected
    else:
        monkeypatch.setenv("GNN_ROOT", str(runtimes["checkout"]))
        expected = str(runtimes["checkout_python"])
        argument = None

    def report(command, **_kwargs):
        # Only the bridge protocol is simulated. No GNN origin/content is qualified.
        payload = {"variables": {"schema": "fixture", "gnn_commit": f"report-from:{command[0]}"}}
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(variables, "subprocess", SimpleNamespace(run=report, DEVNULL=subprocess.DEVNULL))
    payload = variables.build(gnn_runtime=argument)
    assert payload["gnn_commit"] == f"report-from:{expected}"
    assert payload["experiment_engine"] == "cadcad"
    assert payload["experiment_dimension"] == 4
    assert payload["experiment_config_path"] == "configs/alternate.toml"


@pytest.mark.parametrize("outputs", [[], ["one.whl", "two.whl"]])
def test_package_smoke_refuses_missing_or_ambiguous_fresh_output(tmp_path, monkeypatch, outputs):
    stale = tmp_path / "dist" / "zzzz-9999.whl"
    stale.parent.mkdir()
    stale.write_bytes(b"historical stale wheel")
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    installed = []

    def run(command, **_kwargs):
        if command[:2] == ["uv", "build"]:
            output = Path(command[command.index("--out-dir") + 1])
            output.mkdir()
            for name in outputs:
                (output / name).write_bytes(b"fresh")
        elif command[:3] == ["uv", "pip", "install"]:
            installed.append(Path(command[-1]).read_bytes())
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(gate, "_run", run)
    with pytest.raises(ValueError, match="exactly one fresh wheel"):
        gate._package_smoke()
    assert installed == []
    assert stale.read_bytes() == b"historical stale wheel"
    assert list(stale.parent.iterdir()) == [stale]


def test_package_smoke_installs_only_its_own_fresh_bytes_across_invocations(tmp_path, monkeypatch):
    stale = tmp_path / "dist" / "zzzz-9999.whl"
    stale.parent.mkdir()
    stale.write_bytes(b"do not install or delete")
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    directories, installed = [], []

    def run(command, **_kwargs):
        if command[:2] == ["uv", "build"]:
            output = Path(command[command.index("--out-dir") + 1])
            output.mkdir()
            directories.append(output)
            (output / "active_blockference-1.0.0-py3-none-any.whl").write_bytes(b"fresh wheel bytes")
            (output / "unrelated.txt").write_text("not a wheel", encoding="utf-8")
        elif command[:2] == ["uv", "export"]:
            Path(command[-1]).write_text("locked constraints fixture\n", encoding="utf-8")
        elif command[:3] == ["uv", "pip", "install"]:
            wheel = Path(command[-1])
            assert wheel.parent == directories[-1]
            installed.append(wheel.read_bytes())
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(gate, "_run", run)
    gate._package_smoke()
    gate._package_smoke()
    assert installed == [b"fresh wheel bytes", b"fresh wheel bytes"]
    assert directories[0] != directories[1]
    assert all(not path.exists() for path in directories)
    assert stale.read_bytes() == b"do not install or delete"
    assert list(stale.parent.iterdir()) == [stale]


def test_failed_build_never_installs_a_stale_wheel(tmp_path, monkeypatch):
    stale = tmp_path / "dist" / "zzzz-9999.whl"
    stale.parent.mkdir()
    stale.write_bytes(b"retain")
    monkeypatch.setattr(gate, "ROOT", tmp_path)

    def fail_build(command, **_kwargs):
        if command[:2] != ["uv", "build"]:
            raise AssertionError("package processing continued after failed build")
        raise subprocess.CalledProcessError(7, command)

    monkeypatch.setattr(gate, "_run", fail_build)
    with pytest.raises(subprocess.CalledProcessError) as error:
        gate._package_smoke()
    assert error.value.returncode == 7
    assert stale.read_bytes() == b"retain"
