"""Consumer-visible release gate failure reporting."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.release_check as gate

ROOT = Path(__file__).resolve().parents[1]


def test_gnn_arm_runs_the_shared_pin_reader_before_conformance():
    runtime = "/opt/gnn-runtime/bin/python"
    plan = gate._plan("gnn", workdir=Path("<temporary>"), gnn_runtime=runtime)

    assert plan is not None
    assert plan[0] == ["uv", "run", "python", "scripts/export_gnn_spec.py", "--check"]
    assert plan[1] == [
        runtime,
        "-m",
        "gnn.importers",
        "check",
        "--repo",
        ".",
        "--require-pointer",
        "--workflow",
        ".github/workflows/ci.yml",
        "--job",
        "gnn-conformance",
    ]
    assert plan[2] == [
        "uv",
        "run",
        "python",
        "scripts/validate_gnn_spec.py",
        "--gnn-runtime",
        runtime,
        "--json",
    ]


def test_gnn_arm_omits_the_reader_without_a_runtime():
    plan = gate._plan("gnn", workdir=Path("<temporary>"), gnn_runtime=None)

    assert plan is not None
    assert not any("gnn.importers" in " ".join(command) for command in plan)


def test_checkout_runtime_comes_from_the_gnn_checkout(tmp_path, monkeypatch):
    relative = Path("Scripts" if os.name == "nt" else "bin") / (
        "python.exe" if os.name == "nt" else "python"
    )
    python = tmp_path / ".venv" / relative
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")

    monkeypatch.setenv("GNN_ROOT", str(tmp_path))
    assert gate._checkout_runtime() == str(python)

    monkeypatch.delenv("GNN_ROOT")
    assert gate._checkout_runtime() is None


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
