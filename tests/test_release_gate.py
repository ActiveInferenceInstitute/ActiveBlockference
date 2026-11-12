"""Consumer-visible release gate failure reporting."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
