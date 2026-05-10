"""Helpers shared across viz modules — trajectory frame parsing."""

from __future__ import annotations

import ast
from typing import Any

import numpy as np
import pandas as pd


def coerce_cell(value: Any) -> Any:
    """Coerce a cell that may be a Python literal stringified by CSV round-tripping."""
    if isinstance(value, (dict, list, tuple)):
        return value
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def extract_agent_positions(df: pd.DataFrame) -> dict[Any, list[tuple[int, int]]]:
    """Return ``{agent_id: [(y0, x0), (y1, x1), ...]}`` from a trajectory frame."""
    if "env_states" not in df.columns:
        raise ValueError("trajectory frame must contain an 'env_states' column")

    series = df["env_states"].apply(coerce_cell)
    if series.empty:
        return {}

    first = series.iloc[0]
    if not isinstance(first, dict):
        raise ValueError(
            f"'env_states' cells must be dicts, got {type(first).__name__}"
        )

    out: dict[Any, list[tuple[int, int]]] = {k: [] for k in first}
    for cell in series:
        if not isinstance(cell, dict):
            continue
        for k, v in cell.items():
            if isinstance(v, (list, tuple)) and len(v) == 2:
                out.setdefault(k, []).append(tuple(int(c) for c in v))
    return out


def extract_action_history(df: pd.DataFrame) -> dict[Any, list[Any]]:
    """Return ``{agent_id: [a0, a1, ...]}`` from a trajectory frame."""
    if "actions" not in df.columns:
        return {}
    series = df["actions"].apply(coerce_cell)
    out: dict[Any, list[Any]] = {}
    for cell in series:
        if not isinstance(cell, dict):
            continue
        for k, v in cell.items():
            out.setdefault(k, []).append(v)
    return out


def extract_belief_history(df: pd.DataFrame) -> dict[Any, list[np.ndarray]]:
    """Return ``{agent_id: [q_s_0, q_s_1, ...]}`` from a trajectory frame.

    Beliefs that are stored as the empty string (cadCAD initial state) are
    skipped so callers can plot from t≥1.
    """
    if "inferences" not in df.columns:
        return {}
    series = df["inferences"].apply(coerce_cell)
    out: dict[Any, list[np.ndarray]] = {}
    for cell in series:
        if not isinstance(cell, dict):
            continue
        for k, v in cell.items():
            if isinstance(v, (list, tuple)):
                out.setdefault(k, []).append(np.asarray(v, dtype=float))
            elif isinstance(v, np.ndarray):
                out.setdefault(k, []).append(v.astype(float))
            # silently skip empty-string initial entries
    return out


def infer_grid_dimension(df: pd.DataFrame) -> int:
    """Infer ``n`` (the side length) from a trajectory frame.

    Resolution order:
      1. Belief vector size (always ``n*n`` for a square grid) — most reliable.
      2. ``max(coordinate) + 1`` over observed positions — falls back here
         only when no belief history is recorded.
      3. Hard fallback of 3 when neither signal is available.
    """
    beliefs = extract_belief_history(df)
    for history in beliefs.values():
        if history:
            n_from_belief = int(round(np.asarray(history[0]).size ** 0.5))
            if n_from_belief * n_from_belief == np.asarray(history[0]).size:
                return n_from_belief

    positions = extract_agent_positions(df)
    if positions:
        flat = [coord for traj in positions.values() for pair in traj for coord in pair]
        if flat:
            return int(max(flat) + 1)

    return 3
