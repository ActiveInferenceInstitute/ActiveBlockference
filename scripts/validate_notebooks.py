"""Validate tracked notebook hygiene before executing notebooks."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat

ABSOLUTE_PATH_RE = re.compile(r"(?:/Users/|/home/|[A-Za-z]:[\\/])")


def validate_notebook(path: str | Path) -> list[str]:
    """Return contract violations found in one notebook."""

    notebook_path = Path(path)
    errors: list[str] = []
    notebook = nbformat.read(notebook_path, as_version=4)
    cells = notebook.cells
    if not cells or cells[0].cell_type != "markdown":
        errors.append("first cell must be a teaching-summary markdown cell")
    elif len(cells[0].source.split()) < 12:
        errors.append("teaching-summary markdown cell is too short")
    for index, cell in enumerate(cells):
        source = cell.source if isinstance(cell.source, str) else "".join(cell.source)
        if ABSOLUTE_PATH_RE.search(source):
            errors.append(f"cell {index} contains an absolute filesystem path")
        if cell.cell_type == "code":
            if "GRTs" not in notebook_path.parts and "blockference" not in source:
                errors.append(f"code cell {index} does not import the current package API")
            if cell.outputs or cell.execution_count is not None:
                errors.append(f"code cell {index} contains committed execution output")
    return errors


def validate_all(root: str | Path) -> dict[Path, list[str]]:
    """Validate all tracked notebooks below ``root``."""

    root_path = Path(root)
    failures: dict[Path, list[str]] = {}
    for path in sorted(root_path.rglob("*.ipynb")):
        if ".ipynb_checkpoints" in path.parts or ".venv" in path.parts:
            continue
        errors = validate_notebook(path)
        if errors:
            failures[path.relative_to(root_path)] = errors
    return failures


def main() -> int:
    """Validate repository notebooks and print a concise report."""

    root = Path(__file__).resolve().parents[1]
    failures = validate_all(root)
    if failures:
        for path, errors in failures.items():
            for error in errors:
                print(f"ERROR: {path}: {error}")
        return 1
    print("notebook hygiene validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
