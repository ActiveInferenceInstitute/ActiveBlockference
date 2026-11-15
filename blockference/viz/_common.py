"""Strict trajectory parsing shared by visual renderers."""

from __future__ import annotations

import ast
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def atomic_savefig(fig, path: str | Path, **kwargs: Any) -> Path:
    """Render ``fig`` to ``path`` atomically via a sibling temporary file.

    The temporary file keeps the real output suffix (e.g. ``.png``) so
    matplotlib still infers the format, then ``os.replace`` publishes it in one
    step. A crashed render can never leave a partially written final image.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=target.suffix, dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        fig.savefig(temporary, **kwargs)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def coerce_cell(value: Any) -> Any:
    """Decode a nested value written by pandas CSV serialisation."""

    if isinstance(value, (dict, list, tuple, np.ndarray)):
        return value
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value
    return value


def ordered_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows ordered by their unique, non-negative run/time identity."""
    columns = ["run", "timestep", "substep"]
    if df.empty or any(column not in df for column in columns):
        raise ValueError("trajectory requires non-empty run/timestep/substep columns")
    try:
        indices = df[columns].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("trajectory indices must be integers") from exc
    if (
        not np.isfinite(indices).all()
        or (indices < 0).any()
        or (indices[:, 0] < 1).any()
        or not np.equal(indices, np.floor(indices)).all()
        or any(isinstance(value, (bool, np.bool_)) for value in df[columns].to_numpy().flat)
    ):
        raise ValueError("trajectory indices must be non-negative integers with run >= 1")
    ordered = df.assign(**{
        column: indices[:, index].astype(int) for index, column in enumerate(columns)
    }).sort_values(columns, kind="stable")
    if ordered.duplicated(columns).any():
        raise ValueError("trajectory contains duplicate run/timestep/substep identities")
    return ordered


def _mapping_series(df: pd.DataFrame, column: str) -> list[tuple[tuple[int, int, int], dict]]:
    if column not in df.columns:
        raise ValueError(f"trajectory frame must contain {column!r}")
    ordered = ordered_trajectory(df)
    cells = [coerce_cell(value) for value in ordered[column]]
    if any(not isinstance(cell, dict) for cell in cells):
        raise ValueError(f"trajectory column {column!r} must contain mappings")
    indices = ordered[["run", "timestep", "substep"]].itertuples(index=False, name=None)
    return list(zip(indices, cells, strict=True))


def extract_agent_positions(df: pd.DataFrame) -> dict[tuple[int, Any], list[tuple[int, int]]]:
    """Return coordinate histories keyed by ``(run, agent)`` in actual time order."""
    out: dict[tuple[int, Any], list[tuple[int, int]]] = {}
    for (run, _time, _substep), cell in _mapping_series(df, "env_states"):
        for agent_id, value in cell.items():
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"agent {agent_id!r} has an invalid coordinate")
            if any(
                isinstance(component, bool) or not isinstance(component, (int, np.integer))
                for component in value
            ):
                raise ValueError(f"agent {agent_id!r} has a non-integer coordinate")
            coordinate = (int(value[0]), int(value[1]))
            if min(coordinate) < 0:
                raise ValueError(f"agent {agent_id!r} has a negative coordinate")
            out.setdefault((run, agent_id), []).append(coordinate)
    if not out:
        raise ValueError("trajectory contains no agent positions")
    lengths: dict[int, set[int]] = {}
    for (run, _agent), history in out.items():
        lengths.setdefault(run, set()).add(len(history))
    if any(len(values) != 1 for values in lengths.values()):
        raise ValueError("agent trajectories have inconsistent lengths within a run")
    return out


def extract_action_history(df: pd.DataFrame) -> dict[tuple[int, Any], list[Any]]:
    """Return ordered action histories keyed by ``(run, agent)``."""
    out: dict[tuple[int, Any], list[Any]] = {}
    for (run, _time, _substep), cell in _mapping_series(df, "actions"):
        for agent_id, value in cell.items():
            out.setdefault((run, agent_id), []).append(value)
    return out


def _extract_vector_history(df: pd.DataFrame, column: str) -> dict[tuple[int, Any], list[np.ndarray]]:
    out: dict[tuple[int, Any], list[np.ndarray]] = {}
    for (run, _time, _substep), cell in _mapping_series(df, column):
        for agent_id, value in cell.items():
            parsed = coerce_cell(value)
            if parsed is None or (isinstance(parsed, str) and parsed == ""):
                continue
            array = np.asarray(parsed, dtype=float)
            if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
                raise ValueError(f"{column} for agent {agent_id!r} is not a finite vector")
            out.setdefault((run, agent_id), []).append(array)
    return out


def extract_history_steps(df: pd.DataFrame, column: str) -> dict[tuple[int, Any], list[tuple[int, int]]]:
    """Return actual timestep/substep labels aligned with non-empty history cells."""
    out: dict[tuple[int, Any], list[tuple[int, int]]] = {}
    for (run, time, substep), cell in _mapping_series(df, column):
        for agent_id, value in cell.items():
            parsed = coerce_cell(value)
            if parsed is not None and not (isinstance(parsed, str) and parsed == ""):
                out.setdefault((run, agent_id), []).append((time, substep))
    return out


def extract_belief_history(df: pd.DataFrame) -> dict[tuple[int, Any], list[np.ndarray]]:
    """Return persisted posterior vectors by ``(run, agent)``."""

    return _extract_vector_history(df, "inferences")


def extract_efe_history(df: pd.DataFrame) -> dict[tuple[int, Any], list[np.ndarray]]:
    """Return persisted expected-free-energy vectors by ``(run, agent)``."""

    return _extract_vector_history(df, "efe")


def infer_grid_dimension(df: pd.DataFrame) -> int:
    """Infer a square grid side length from beliefs or positions."""

    beliefs = extract_belief_history(df)
    sizes = {int(history[0].size) for history in beliefs.values() if history}
    for size in sizes:
        side = int(round(size**0.5))
        if side * side == size:
            return side
    positions = extract_agent_positions(df)
    max_coordinate = max(component for history in positions.values() for pair in history for component in pair)
    side = max_coordinate + 1
    if side < 2:
        raise ValueError("grid dimension must be at least 2")
    return side


__all__ = [
    "atomic_savefig",
    "coerce_cell",
    "extract_action_history",
    "extract_agent_positions",
    "extract_belief_history",
    "extract_efe_history",
    "extract_history_steps",
    "infer_grid_dimension",
    "ordered_trajectory",
]
