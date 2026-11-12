"""Print the upstream ``gnn.importers`` reading of this repository's GNN pin.

``gnn.importers`` is the estate's single reader for a GNN pin (discovery, shape
normalization, pointer checks and the ``gnn-importer-variables/v1`` map). This
module is the bridge that calls it from a Python process: it runs in an
interpreter where ``gnn`` is importable, and prints the normalized pin plus the
GNN-derived manuscript variables as JSON. Neither this repository nor its
scripts re-implement the pin shape.

Run it with the pinned runtime (``GNN_RUNTIME_PYTHON``), or with a pinned source
checkout first on ``PYTHONPATH``. ``scripts/validate_gnn_spec.py`` and
``scripts/manuscript_variables.py`` invoke it; the release gate's ``gnn`` arm
calls ``python -m gnn.importers check`` directly and does not need this bridge.

Usage (normally invoked by those scripts, not by hand):

    GNN_RUNTIME_PYTHON/bin/python scripts/gnn_importer_report.py --repo .
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from gnn.importers import find_pin, importer_variables, read_pin


def report(
    repo_root: Path, *, job: str | None = None, job_name: str | None = None
) -> dict[str, Any]:
    """Return the shared reader's pin record and manuscript variables."""

    pin_path = find_pin(repo_root)
    if pin_path is None:
        raise SystemExit(f"no GNN pin file found under {repo_root}")
    pin = read_pin(pin_path)
    parser_files = pin.raw.get("parser_files")
    return {
        "pin": {
            "path": pin.path.relative_to(repo_root).as_posix(),
            "schema": pin.schema,
            "repository": pin.repository,
            "tag": pin.tag,
            "commit": pin.commit,
            "recorded": pin.recorded,
            "spec_files": list(pin.spec_files),
            "parser_files": parser_files if isinstance(parser_files, dict) else {},
        },
        "variables": importer_variables(pin, repo_root=repo_root, job=job, job_name=job_name),
    }


def main(argv: list[str] | None = None) -> int:
    """Print the report for one repository checkout."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--job", help="conformance job key the pointer and variables name")
    parser.add_argument("--job-name", help="display name of that job")
    args = parser.parse_args(argv)
    payload = report(args.repo.resolve(), job=args.job, job_name=args.job_name)
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
