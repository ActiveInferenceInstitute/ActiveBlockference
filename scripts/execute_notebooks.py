"""Parse and execute every tracked notebook from the repository root."""

from __future__ import annotations

import tempfile
from pathlib import Path

from blockference.notebook_inventory import execute_notebooks


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="activeblockference-notebooks-") as workspace:
        notebooks = execute_notebooks(root, workspace=workspace)
        for path in notebooks:
            print(path.relative_to(root))
        print(f"executed {len(notebooks)} tracked notebooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
