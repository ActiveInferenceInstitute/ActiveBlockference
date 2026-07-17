"""Strict trajectory parsing shared by visual renderers."""

from __future__ import annotations

import ast
from typing import Any

import numpy as np
import pandas as pd


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


def _mapping_series(df: pd.DataFrame, column: str) -> list[dict[Any, Any]]:
    if column not in df.columns:
        raise ValueError(f"trajectory frame must contain {column!r}")
    cells = [coerce_cell(value) for value in df[column]]
    if any(not isinstance(cell, dict) for cell in cells):
        raise ValueError(f"trajectory column {column!r} must contain mappings")
    return cells


def extract_agent_positions(df: pd.DataFrame) -> dict[Any, list[tuple[int, int]]]:
    """Return ``{agent_id: [(y, x), ...]}`` from a trajectory frame."""

    cells = _mapping_series(df, "env_states")
    if not cells:
        raise ValueError("trajectory frame is empty")
    out: dict[Any, list[tuple[int, int]]] = {}
    for cell in cells:
        for agent_id, value in cell.items():
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"agent {agent_id!r} has an invalid coordinate")
            if any(isinstance(component, bool) or not isinstance(component, (int, np.integer)) for component in value):
                raise ValueError(f"agent {agent_id!r} has a non-integer coordinate")
            coordinate = (int(value[0]), int(value[1]))
            if min(coordinate) < 0:
                raise ValueError(f"agent {agent_id!r} has a negative coordinate")
            out.setdefault(agent_id, []).append(coordinate)
    if not out:
        raise ValueError("trajectory contains no agent positions")
    lengths = {len(history) for history in out.values()}
    if len(lengths) != 1:
        raise ValueError("agent trajectories have inconsistent lengths")
    return out


def extract_action_history(df: pd.DataFrame) -> dict[Any, list[Any]]:
    """Return ``{agent_id: [action, ...]}`` from a trajectory frame."""

    cells = _mapping_series(df, "actions")
    out: dict[Any, list[Any]] = {}
    for cell in cells:
        for agent_id, value in cell.items():
            out.setdefault(agent_id, []).append(value)
    return out


def _extract_vector_history(df: pd.DataFrame, column: str) -> dict[Any, list[np.ndarray]]:
    cells = _mapping_series(df, column)
    out: dict[Any, list[np.ndarray]] = {}
    for cell in cells:
        for agent_id, value in cell.items():
            parsed = coerce_cell(value)
            if parsed is None or (isinstance(parsed, str) and parsed == ""):
                continue
            array = np.asarray(parsed, dtype=float)
            if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
                raise ValueError(f"{column} for agent {agent_id!r} is not a finite vector")
            out.setdefault(agent_id, []).append(array)
    return out


def extract_belief_history(df: pd.DataFrame) -> dict[Any, list[np.ndarray]]:
    """Return persisted posterior vectors by agent."""

    return _extract_vector_history(df, "inferences")


def extract_efe_history(df: pd.DataFrame) -> dict[Any, list[np.ndarray]]:
    """Return persisted expected-free-energy vectors by agent."""

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
    "coerce_cell",
    "extract_action_history",
    "extract_agent_positions",
    "extract_belief_history",
    "extract_efe_history",
    "infer_grid_dimension",
]
