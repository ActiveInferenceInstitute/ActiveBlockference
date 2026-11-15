"""Select the separate GNN interpreter without importing its NumPy environment."""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["resolve_gnn_runtime"]


def resolve_gnn_runtime(
    explicit: str | Path | None = None,
    *,
    checkout: str | Path | None = None,
    python: str | Path | None = None,
    required: bool = True,
) -> str | None:
    """Select CLI runtime, runtime environment, legacy Python, then checkout venv.

    Explicitly selected missing files fail without substitution. Keep the venv
    launcher path rather than resolving its symlink to a base interpreter.
    """
    selected = explicit
    if selected is None:
        selected = os.environ.get("GNN_RUNTIME_PYTHON") or (
            python if python is not None else os.environ.get("GNN_PYTHON") or None
        )
    root = checkout or os.environ.get("GNN_ROOT")
    if selected is None and root:
        relative = Path("Scripts/python.exe" if os.name == "nt" else "bin/python")
        selected = Path(root) / ".venv" / relative
    if selected is None:
        if required:
            raise FileNotFoundError(
                "select --gnn-runtime/GNN_RUNTIME_PYTHON or a GNN_ROOT with its own .venv"
            )
        return None
    candidate = Path(selected).expanduser().absolute()
    if not candidate.is_file():
        raise FileNotFoundError(f"selected GNN interpreter does not exist: {candidate}")
    return str(candidate)
