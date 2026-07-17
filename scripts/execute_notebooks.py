"""Parse and execute every tracked notebook from the repository root."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    for path in sorted(root.rglob("*.ipynb")):
        if ".venv" in path.parts or ".git" in path.parts:
            continue
        notebook = nbformat.read(path, as_version=4)
        NotebookClient(notebook, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(root)}}).execute()
        print(path.relative_to(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
