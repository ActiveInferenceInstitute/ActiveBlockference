"""Normalized typed trajectory schema for deterministic backend parity.

The simulation backends (radCAD and cadCAD) store per-step diagnostics as nested
Python objects, which pandas stringifies when persisted to CSV. Comparing the
raw stringified forms can hide type or precision drift. This module parses a
trajectory frame into an ordered list of frozen :class:`TrajectoryRecord` values
so that two runs — or a run and its persisted round-trip — can be compared on
the typed values themselves (coordinates, integer indices, and float vectors).

The records are sorted deterministically by ``(run, timestep, substep, agent)``
so the comparison is insensitive to the backend's internal row ordering.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["TrajectoryRecord", "parse_trajectory_records"]


def _literal(value: Any) -> Any:
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


def _scalar(value: Any) -> int | None:
    parsed = _literal(value)
    if isinstance(parsed, bool) or not isinstance(parsed, (int, np.integer)):
        return None
    return int(parsed)


def _vector(value: Any) -> tuple[float, ...] | None:
    parsed = _literal(value)
    if parsed is None:
        return None
    try:
        array = np.asarray(parsed, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        return None
    return tuple(float(item) for item in array)


def _coordinate(value: Any) -> tuple[int, int] | None:
    parsed = _literal(value)
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 2:
        return None
    if any(
        isinstance(component, (bool, np.bool_)) or not isinstance(component, (int, np.integer))
        for component in parsed
    ):
        return None
    return int(parsed[0]), int(parsed[1])


@dataclass(frozen=True)
class TrajectoryRecord:
    """Typed per-(run, timestep, agent) diagnostic record.

    Every vector is stored as a tuple of finite floats and every index as an
    integer, so equality between two records means the underlying values agree
    exactly — independent of how the backend or the CSV writer stringified them.
    """

    run: int
    substep: int
    timestep: int
    agent_id: Any
    env_state: tuple[int, int] | None
    action: int | None
    obs_idx: int | None
    posterior: tuple[float, ...] | None
    prior: tuple[float, ...] | None
    efe: tuple[float, ...] | None
    efe_epistemic: tuple[float, ...] | None
    efe_pragmatic: tuple[float, ...] | None
    q_pi: tuple[float, ...] | None
    p_u: tuple[float, ...] | None


def parse_trajectory_records(df: pd.DataFrame) -> list[TrajectoryRecord]:
    """Flatten a trajectory frame into deterministically ordered typed records."""
    records: list[TrajectoryRecord] = []
    if df.empty:
        return records
    for _, row in df.iterrows():
        env = _literal(row.get("env_states"))
        if not isinstance(env, dict):
            continue
        run = _scalar(row.get("run"))
        substep = _scalar(row.get("substep"))
        timestep = _scalar(row.get("timestep"))
        actions = _literal(row.get("actions")) or {}
        inferences = _literal(row.get("inferences")) or {}
        priors = _literal(row.get("priors")) or {}
        efes = _literal(row.get("efe")) or {}
        efes_eps = _literal(row.get("efe_epistemic")) or {}
        efes_prag = _literal(row.get("efe_pragmatic")) or {}
        q_pis = _literal(row.get("q_pi")) or {}
        p_us = _literal(row.get("p_u")) or {}
        obs_idxs = _literal(row.get("obs_idx")) or {}
        for agent_id in env:
            records.append(
                TrajectoryRecord(
                    run=-1 if run is None else run,
                    substep=-1 if substep is None else substep,
                    timestep=-1 if timestep is None else timestep,
                    # Persistence intentionally stringifies agent keys for JSON
                    # compatibility, so canonicalize here for type-stable parity.
                    agent_id=str(agent_id),
                    env_state=_coordinate(env.get(agent_id)),
                    action=_scalar(actions.get(agent_id)),
                    obs_idx=_scalar(obs_idxs.get(agent_id)),
                    posterior=_vector(inferences.get(agent_id)),
                    prior=_vector(priors.get(agent_id)),
                    efe=_vector(efes.get(agent_id)),
                    efe_epistemic=_vector(efes_eps.get(agent_id)),
                    efe_pragmatic=_vector(efes_prag.get(agent_id)),
                    q_pi=_vector(q_pis.get(agent_id)),
                    p_u=_vector(p_us.get(agent_id)),
                )
            )
    records.sort(key=lambda r: (r.run, r.timestep, r.substep, repr(r.agent_id)))
    return records
