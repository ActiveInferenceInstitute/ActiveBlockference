"""Derive the manuscript variables from repository sources.

Manuscript prose uses ``{{token}}`` placeholders; the declared values come from
this generated file rather than from hand-copied literals. The values are read
from ``pyproject.toml`` (Python floor and package version), the executable
manuscript configuration, the CI workflow, and the upstream ``gnn.importers``
reader for the ``gnn_*`` keys. The reader needs the pinned GNN runtime
(``GNN_RUNTIME_PYTHON``) because its distribution resolves NumPy 2.x while this
package resolves NumPy 1.x.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
VARIABLES_PATH = ROOT / "docs" / "_generated" / "manuscript-variables.json"
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "ci.yml"
REPORT = ROOT / "scripts" / "gnn_importer_report.py"
SCHEMA = "manuscript-variables/v1"
MANUSCRIPT_CONFIG_PATH = ROOT / "docs" / "manuscript" / "config.yaml"


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


def _gnn_interpreter() -> str:
    """Return the interpreter that has the pinned GNN distribution installed."""

    runtime = os.environ.get("GNN_RUNTIME_PYTHON")
    if runtime and Path(runtime).is_file():
        return runtime
    return sys.executable


def _gnn_variables(job_key: str, job_name: str) -> dict[str, str]:
    """Return the ``gnn_*`` keys from ``gnn.importers.importer_variables``."""

    completed = subprocess.run(
        [
            _gnn_interpreter(),
            str(REPORT),
            "--repo", str(ROOT),
            "--job", job_key,
            "--job-name", job_name,
        ],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        raise ValueError(
            "the shared gnn.importers reader could not produce the gnn_* variables: "
            + (" | ".join(detail[-3:]) if detail else "no output")
            + "; set GNN_RUNTIME_PYTHON to the pinned GNN runtime"
        )
    variables = dict(json.loads(completed.stdout)["variables"])
    variables.pop("schema", None)  # gnn-importer-variables/v1 names the map, not this file
    return variables


def build() -> dict[str, object]:
    """Return the declared manuscript variables and their sources."""

    project = _project()
    job_key, job_name = _gnn_job()
    gnn = _gnn_variables(job_key, job_name)
    metadata = yaml.safe_load(MANUSCRIPT_CONFIG_PATH.read_text(encoding="utf-8"))
    config_path = metadata["experiment"]["config_path"]
    config = yaml.safe_load((ROOT / config_path).read_text(encoding="utf-8"))
    experiment_values = {
        "experiment_config_path": config_path,
        "experiment_engine": config["engine"],
        "experiment_dimension": config["grid"]["dimension"],
        "experiment_agents": config["simulation"]["n_agents"],
        "experiment_planning_length": config["grid"]["planning_length"],
        "experiment_timesteps": config["simulation"]["timesteps"],
        "experiment_seed": config["seed"],
        "experiment_affordance_count": len(config["grid"]["affordances"]),
        "experiment_affordances": ", ".join(f"`{value}`" for value in config["grid"]["affordances"]),
    }
    experiment_sources = {
        "experiment_config_path": "docs/manuscript/config.yaml experiment.config_path",
        "experiment_engine": f"{config_path} engine",
        "experiment_dimension": f"{config_path} grid.dimension",
        "experiment_agents": f"{config_path} simulation.n_agents",
        "experiment_planning_length": f"{config_path} grid.planning_length",
        "experiment_timesteps": f"{config_path} simulation.timesteps",
        "experiment_seed": f"{config_path} seed",
        "experiment_affordance_count": f"{config_path} len(grid.affordances)",
        "experiment_affordances": f"{config_path} grid.affordances",
    }
    return {
        "schema": SCHEMA,
        "python_requires": _python_requires(str(project["requires-python"])),
        "package_version": str(project["version"]),
        "gate_entry_point": "scripts/release_check.py",
        **gnn,
        **experiment_values,
        "sources": {
            "python_requires": "pyproject.toml [project].requires-python",
            "package_version": "pyproject.toml [project].version",
            "gate_entry_point": "scripts/release_check.py",
            "gnn_pin_path": "gnn.importers.importer_variables pin_path",
            "gnn_pin_schema": "gnn.importers.importer_variables pin_schema",
            "gnn_repository": "gnn.importers.importer_variables repository",
            "gnn_tag": "gnn.importers.importer_variables tag",
            "gnn_commit": "gnn.importers.importer_variables commit",
            "gnn_job_key": ".github/workflows/ci.yml GNN job key",
            "gnn_job_name": ".github/workflows/ci.yml GNN job name",
            **experiment_sources,
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
