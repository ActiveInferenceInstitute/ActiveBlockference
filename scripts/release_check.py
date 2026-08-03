"""Run the canonical local release gate for ActiveBlockference."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], *, cwd: Path = ROOT, capture: bool = False) -> subprocess.CompletedProcess[str]:
    """Run one release-gate command with a consistent environment."""

    print("$", " ".join(command), flush=True)
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=capture,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )


def _changed_paths() -> set[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only"], cwd=ROOT, check=True, text=True, capture_output=True
    )
    return {line for line in result.stdout.splitlines() if line}


def _assert_json(command: list[str]) -> dict[str, object]:
    result = _run(command, capture=True)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"command emitted no JSON: {' '.join(command)}")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"command did not end in JSON: {lines[-1]!r}") from exc
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise ValueError(f"command returned a non-passing JSON verdict: {payload!r}")
    return payload


def _package_smoke() -> None:
    """Build a wheel and import it from a clean temporary virtualenv."""

    _run(["uv", "build"])
    wheels = sorted((ROOT / "dist").glob("*.whl"))
    if not wheels:
        raise FileNotFoundError("uv build produced no wheel")
    with tempfile.TemporaryDirectory(prefix="activeblockference-package-") as directory:
        venv = Path(directory) / "venv"
        _run(["uv", "venv", "--python", sys.executable, str(venv)])
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        _run(["uv", "pip", "install", "--python", str(python), str(wheels[-1])])
        _run(
            [
                str(python),
                "-c",
                "from blockference import GridWorld; "
                "assert GridWorld(2, {'agent': (0, 0)}).current_state == {'agent': (0, 0)}",
            ]
        )


def main() -> int:
    """Execute lint, tests, artefact, publication, and package gates."""

    _run(["uv", "run", "ruff", "check", "blockference", "tests", "GRTs", "scripts"])
    _run(["uv", "run", "python", "scripts/check_docs_links.py"])
    _run(
        [
            "uv",
            "run",
            "pytest",
            "--cov=blockference",
            "--cov-report=term",
            "--cov-report=xml",
            "--cov-fail-under=90",
        ]
    )
    _run(["uv", "run", "mypy", "blockference"])
    _run(["uv", "run", "python", "scripts/validate_notebooks.py"])
    _run(["uv", "run", "python", "scripts/execute_notebooks.py"])
    before_generated = _changed_paths()
    with tempfile.TemporaryDirectory(prefix="activeblockference-release-") as directory:
        output_root = Path(directory) / "output"
        _assert_json(
            [
                "uv",
                "run",
                "blockference",
                "pipeline",
                "--config",
                "configs/smoke.yml",
                "--output-root",
                str(output_root),
                "--run-name",
                "release_smoke",
            ]
        )
        _assert_json(
            [
                "uv",
                "run",
                "blockference",
                "validation",
                "--run-dir",
                str(output_root / "release_smoke"),
            ]
        )
        _assert_json(
            [
                "uv",
                "run",
                "blockference",
                "run",
                "--config",
                "configs/smoke.yml",
                "--output",
                str(Path(directory) / "run.csv"),
            ]
        )
    _run(["uv", "run", "python", "scripts/validate_manuscript.py"])
    _run(["uv", "run", "python", "scripts/build_manuscript.py"])
    _run(["uv", "run", "python", "scripts/build_manuscript.py", "--check"])
    after_generated = _changed_paths()
    new_changes = sorted(after_generated - before_generated)
    if new_changes:
        raise RuntimeError("generated files drifted during release check: " + ", ".join(new_changes))
    _package_smoke()
    print("release gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
