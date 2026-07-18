"""Parse and execute every tracked notebook from the repository root."""

from __future__ import annotations

import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    try:
        from .validate_notebooks import validate_all
    except ImportError:
        from validate_notebooks import validate_all

    failures = validate_all(root)
    if failures:
        raise ValueError(f"notebook hygiene violations: {failures}")
    with tempfile.TemporaryDirectory(prefix="activeblockference-notebooks-") as workspace:
        workspace_path = Path(workspace)
        for path in sorted(root.rglob("*.ipynb")):
            if ".venv" in path.parts or ".git" in path.parts or ".ipynb_checkpoints" in path.parts:
                continue
            notebook = nbformat.read(path, as_version=4)
            NotebookClient(
                notebook,
                timeout=120,
                kernel_name="python3",
                resources={"metadata": {"path": str(workspace_path)}},
            ).execute()
            print(path.relative_to(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
