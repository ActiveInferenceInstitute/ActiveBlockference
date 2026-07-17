"""Parse and execute every repository notebook."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient


def test_notebooks_parse_and_execute():
    root = Path(__file__).parents[1]
    for path in sorted(root.rglob("*.ipynb")):
        if ".venv" in path.parts:
            continue
        notebook = nbformat.read(path, as_version=4)
        NotebookClient(notebook, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(root)}}).execute()
