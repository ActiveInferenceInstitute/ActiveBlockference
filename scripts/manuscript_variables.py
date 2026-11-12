"""Derive the manuscript variables from repository sources.

Manuscript prose uses ``{{token}}`` placeholders; the declared values come from
this generated file rather than from hand-copied literals. The values are read
from ``pyproject.toml`` (Python floor and package version), the canonical GNN
pin, and the CI workflow (the conformance job key and display name).
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
VARIABLES_PATH = ROOT / "docs" / "_generated" / "manuscript-variables.json"
PIN_PATH = ROOT / "specs" / "gnn-pin.json"
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "ci.yml"
SCHEMA = "manuscript-variables/v1"


def _project() -> dict[str, object]:
    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return payload["project"]


def _python_requires(value: str) -> str:
    spec = value.strip()
    if not spec.startswith(">="):
        raise ValueError(f"unsupported requires-python specifier {value!r}")
    return spec[2:] + "+"


def _gnn_job() -> tuple[str, str]:
    workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    for key, job in workflow["jobs"].items():
        if "gnn" in key:
            return key, job.get("name", key)
    raise ValueError("no GNN conformance job in .github/workflows/ci.yml")


def build() -> dict[str, object]:
    """Return the declared manuscript variables and their sources."""

    project = _project()
    pin = json.loads(PIN_PATH.read_text(encoding="utf-8"))
    job_key, job_name = _gnn_job()
    return {
        "schema": SCHEMA,
        "python_requires": _python_requires(str(project["requires-python"])),
        "package_version": str(project["version"]),
        "gate_entry_point": "scripts/release_check.py",
        "gnn_pin_path": "specs/gnn-pin.json",
        "gnn_pin_schema": str(pin["schema"]),
        "gnn_repository": str(pin["repository"]),
        "gnn_tag": str(pin["tag"]),
        "gnn_commit": str(pin["commit"]),
        "gnn_job_key": job_key,
        "gnn_job_name": job_name,
        "sources": {
            "python_requires": "pyproject.toml [project].requires-python",
            "package_version": "pyproject.toml [project].version",
            "gate_entry_point": "scripts/release_check.py",
            "gnn_pin_path": "specs/gnn-pin.json",
            "gnn_pin_schema": "specs/gnn-pin.json schema",
            "gnn_repository": "specs/gnn-pin.json repository",
            "gnn_tag": "specs/gnn-pin.json tag",
            "gnn_commit": "specs/gnn-pin.json commit",
            "gnn_job_key": ".github/workflows/ci.yml GNN job key",
            "gnn_job_name": ".github/workflows/ci.yml GNN job name",
        },
    }


def variables() -> dict[str, object]:
    """Return the committed variables, failing when the file is absent."""

    if not VARIABLES_PATH.is_file():
        raise FileNotFoundError(
            f"missing {VARIABLES_PATH.relative_to(ROOT)}; run scripts/manuscript_variables.py"
        )
    return json.loads(VARIABLES_PATH.read_text(encoding="utf-8"))


def render(payload: dict[str, object]) -> str:
    """Render the variables file content."""

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write the variables file, or verify it with ``--check``."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail when the file is stale")
    args = parser.parse_args(argv)
    try:
        content = render(build())
    except (OSError, ValueError, KeyError, yaml.YAMLError) as exc:
        print(f"manuscript variables failed: {exc}", file=sys.stderr)
        return 1
    if args.check:
        if not VARIABLES_PATH.is_file():
            print(
                f"manuscript variables failed: missing {VARIABLES_PATH.relative_to(ROOT)}",
                file=sys.stderr,
            )
            return 1
        if VARIABLES_PATH.read_text(encoding="utf-8") != content:
            print(
                "manuscript variables drift: run uv run python scripts/manuscript_variables.py",
                file=sys.stderr,
            )
            return 1
        print("manuscript variables check passed")
        return 0
    VARIABLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    VARIABLES_PATH.write_text(content, encoding="utf-8")
    print(f"wrote {VARIABLES_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
