"""Consumer-visible release gate failure reporting."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import scripts.release_check as gate
from blockference.gnn_source import BOOTSTRAP_SHA256, SOURCE_MANIFEST_SHA256

ROOT = Path(__file__).resolve().parents[1]


def test_checkout_runtime_comes_from_the_gnn_checkout(tmp_path, monkeypatch):
    relative = Path("Scripts" if os.name == "nt" else "bin") / (
        "python.exe" if os.name == "nt" else "python"
    )
    python = tmp_path / ".venv" / relative
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")

    monkeypatch.setenv("GNN_ROOT", str(tmp_path))
    monkeypatch.delenv("GNN_RUNTIME_PYTHON", raising=False)
    monkeypatch.delenv("GNN_PYTHON", raising=False)
    assert gate.resolve_gnn_runtime() == str(python)

    monkeypatch.delenv("GNN_ROOT")
    assert gate.resolve_gnn_runtime(required=False) is None


def test_gnn_arm_without_runtime_cannot_write_a_passing_receipt(tmp_path):
    environment = dict(os.environ, UV_OFFLINE="1", UV_NO_SYNC="1")
    for name in ("GNN_ROOT", "GNN_RUNTIME_PYTHON", "GNN_PYTHON"):
        environment.pop(name, None)
    receipt = tmp_path / "gate.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "release_check.py"),
            "--arm", "gnn", "--timings", str(receipt), "--json",
        ],
        cwd=ROOT,
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "gnn validation not run" in result.stderr
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["arms"]["gnn"]["status"] == "failed"
    summary = next(line for line in result.stdout.splitlines() if line.startswith('{"arms":'))
    assert json.loads(summary)["ok"] is False


def test_render_receipt_identifies_the_produced_pdf(tmp_path, monkeypatch):
    pdf = tmp_path / "manuscript.pdf"
    content = b"%PDF-1.4\nrender receipt regression\n"
    receipt = tmp_path / "gate.json"
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setitem(gate.RENDER_TARGETS, "pdf", pdf)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-c", "user.name=Gate Test", "-c", "user.email=gate@example.invalid",
         "commit", "--allow-empty", "-qm", "source"],
        cwd=tmp_path, check=True,
    )
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True,
    ).strip()

    def render(command):
        pdf.write_bytes(content)

    monkeypatch.setattr(gate, "_run", render)
    assert gate.main(["--arm", "render", "--timings", str(receipt)]) == 0
    recorded = json.loads(receipt.read_text(encoding="utf-8"))["arms"]["render"]
    assert recorded["status"] == "passed"
    assert recorded["artifact"] == {
        "path": "manuscript.pdf",
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "source_commit": commit,
    }


def test_render_provenance_requires_the_pdf(tmp_path, monkeypatch):
    monkeypatch.setitem(gate.RENDER_TARGETS, "pdf", tmp_path / "absent.pdf")
    with pytest.raises(FileNotFoundError):
        gate._render_artifact()


def test_workflow_delivers_reviewed_resources_before_release():
    """Keep immutable resource delivery and release-environment forwarding together."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["gnn-conformance"]["steps"]
    resources = next(step for step in steps if step.get("id") == "resources")
    release = next(step for step in steps if "--arm all" in step.get("run", ""))
    assert steps.index(resources) < steps.index(release)
    script = resources["run"]
    assert "66d491f86895443cf588011cc6f40c37b8bae8fd/interop" in script
    assert f"{BOOTSTRAP_SHA256}  gnn_bootstrap.py" in script
    assert f"{SOURCE_MANIFEST_SHA256}  v5.1.0-source-manifest.json" in script
    assert script.count("sha256sum --check --strict") == 2
    assert script.index("sha256sum --check --strict") < script.index('>> "$GITHUB_OUTPUT"')
    assert release["env"]["GNN_BOOTSTRAP"] == "${{ steps.resources.outputs.bootstrap }}"
    assert release["env"]["GNN_SOURCE_MANIFEST"] == "${{ steps.resources.outputs.manifest }}"
    assert release["env"]["GNN_RUNTIME_PYTHON"] == "${{ steps.runtime.outputs.python }}"


def test_failed_captured_command_keeps_real_diagnostics_and_exit(tmp_path, capsys):
    command = [
        sys.executable, "-c",
        "import sys; print('real failure detail'); "
        "print('real stderr detail', file=sys.stderr); sys.exit(7)",
    ]
    with pytest.raises(subprocess.CalledProcessError) as caught:
        gate._run(command, cwd=tmp_path, capture=True)
    assert caught.value.returncode == 7
    captured = capsys.readouterr()
    assert "real failure detail" in captured.out
    assert "real stderr detail" in captured.err
