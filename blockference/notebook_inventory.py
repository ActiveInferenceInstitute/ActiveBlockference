"""Tracked-source notebook selection, hygiene and isolated-workspace execution."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

__all__ = ["execute_notebooks", "tracked_notebooks", "validate_all", "validate_notebook"]

ABSOLUTE_PATH_RE = re.compile(r"(?:/Users/|/home/|[A-Za-z]:[\\/])")
FORBIDDEN_COMPONENTS = frozenset({
    ".git", ".ipynb_checkpoints", ".venv", "venv", "env", "cad", "build", "dist",
    "output", "results", "node_modules",
})


def _git(root: Path, *arguments: str, data: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update({"GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"})
    return subprocess.run(
        ["git", "--no-optional-locks", "-c", "core.fsmonitor=false", "-C", str(root), *arguments],
        capture_output=True, check=False, env=environment,
        input=data, stdin=subprocess.DEVNULL if data is None else None,
    )


def tracked_notebooks(root: str | Path) -> tuple[Path, ...]:
    """Return only present, non-ignored tracked notebooks at an exact Git root.

    A worktree .git file is valid. Git failure, missing tracked files and symlink
    escapes are errors; none is an empty inventory or a reason to scan the disk.
    """
    repository = Path(root).resolve(strict=True)
    top = _git(repository, "rev-parse", "--show-toplevel")
    if top.returncode or Path(os.fsdecode(top.stdout).removesuffix("\n")).resolve() != repository:
        raise ValueError(f"not an exact Git repository root: {repository}")
    tracked = _git(repository, "ls-files", "--cached", "-z", "--", "*.ipynb")
    if tracked.returncode:
        raise RuntimeError("cannot read tracked notebooks: " + os.fsdecode(tracked.stderr))
    names = set(name for name in tracked.stdout.split(b"\0") if name)
    if not names:
        return ()
    ignored = _git(
        repository, "check-ignore", "--no-index", "--stdin", "-z",
        data=b"\0".join(sorted(names)) + b"\0",
    )
    if ignored.returncode not in (0, 1):
        raise RuntimeError("cannot classify ignored notebooks: " + os.fsdecode(ignored.stderr))
    excluded = set(ignored.stdout.split(b"\0"))
    selected = []
    for name in sorted(names - excluded):
        relative = Path(os.fsdecode(name))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid tracked notebook path: {relative}")
        if any(part.lower() in FORBIDDEN_COMPONENTS for part in relative.parts):
            continue
        candidate = repository / relative
        if candidate.is_symlink() or not candidate.resolve().is_relative_to(repository):
            raise ValueError(f"tracked notebook is a symlink or escapes the repository: {relative}")
        if not candidate.is_file():
            raise FileNotFoundError(f"tracked notebook is missing: {relative}")
        selected.append(candidate)
    return tuple(selected)


def validate_notebook(path: str | Path) -> list[str]:
    """Return teaching-summary, source-path, API and committed-output violations."""
    import nbformat

    notebook_path = Path(path)
    notebook = nbformat.read(notebook_path, as_version=4)
    errors = []
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


def _validate_selection(root: Path, notebooks: tuple[Path, ...]) -> dict[Path, list[str]]:
    failures = {}
    for path in notebooks:
        errors = validate_notebook(path)
        if errors:
            failures[path.relative_to(root)] = errors
    return failures


def validate_all(root: str | Path) -> tuple[tuple[Path, ...], dict[Path, list[str]]]:
    """Return the frozen tracked selection and its violations; empty is explicit."""
    repository = Path(root).resolve(strict=True)
    notebooks = tracked_notebooks(repository)
    return notebooks, _validate_selection(repository, notebooks)


def execute_notebooks(root: str | Path, *, workspace: str | Path) -> tuple[Path, ...]:
    """Validate and execute one frozen tracked selection in the supplied workspace."""
    notebooks, failures = validate_all(root)
    if failures:
        raise ValueError(f"notebook hygiene violations: {failures}")
    if not notebooks:
        return notebooks
    import nbformat
    from nbclient import NotebookClient

    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        NotebookClient(
            notebook, timeout=120, kernel_name="python3",
            resources={"metadata": {"path": str(Path(workspace).resolve())}},
        ).execute()
    return notebooks
