"""Serialisation helpers — write configs, dataframes, and summaries to disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from blockference.config import ExperimentConfig
from blockference.io.layout import RunPaths

__all__ = [
    "persist_config",
    "persist_dataframe",
    "persist_generative_model",
    "persist_generative_model_npz",
    "persist_per_step_records",
    "persist_policies",
    "persist_summary",
]


def persist_config(cfg: ExperimentConfig, paths: RunPaths) -> Path:
    """Write the run's config back to disk so the artefacts are self-describing."""
    paths.ensure()
    with paths.config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(_to_jsonable(cfg.to_dict()), fh, sort_keys=False)
    return paths.config_path


def persist_dataframe(df: pd.DataFrame, paths: RunPaths) -> dict[str, Path]:
    """Write the trajectory frame as CSV (always) and Parquet (if pyarrow)."""
    paths.ensure()
    written: dict[str, Path] = {}
    df.to_csv(paths.trajectory_csv, index=False)
    written["csv"] = paths.trajectory_csv
    try:
        df.to_parquet(paths.trajectory_parquet, index=False)
        written["parquet"] = paths.trajectory_parquet
    except (ImportError, ValueError):
        # parquet writer (pyarrow/fastparquet) not installed or schema incompatible
        pass
    return written


def persist_summary(summary: dict[str, Any], paths: RunPaths) -> Path:
    """Write the per-run summary JSON (number of agents, final positions, ...)."""
    paths.ensure()
    with paths.summary_json.open("w", encoding="utf-8") as fh:
        json.dump(_to_jsonable(summary), fh, indent=2, default=str)
    return paths.summary_json


def persist_generative_model(agents: dict, paths: RunPaths) -> Path:
    """Persist each agent's A/B/C/D/E matrices to a single JSON file.

    Parameters
    ----------
    agents :
        ``{agent_id: ActiveGridference}`` mapping. Only the public matrices
        are serialised; per-step state is captured in :func:`persist_per_step_records`.
    """
    paths.ensure()
    payload: dict[str, Any] = {}
    for agent_id, agent in agents.items():
        payload[str(agent_id)] = {
            "n_states": int(getattr(agent, "n_states", 0)),
            "n_observations": int(getattr(agent, "n_observations", 0)),
            "border": int(getattr(agent, "border", 0)),
            "policy_len": int(getattr(agent, "policy_len", 0)),
            "E": list(getattr(agent, "E", [])),
            "A": _to_jsonable(getattr(agent, "A", None)),
            "B": _to_jsonable(getattr(agent, "B", None)),
            "C": _to_jsonable(getattr(agent, "C", None)),
            "D": _to_jsonable(getattr(agent, "D", None)),
        }
    with paths.matrices_json.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return paths.matrices_json


def persist_per_step_records(records: list[dict[str, Any]], paths: RunPaths) -> Path:
    """Write per-step / per-agent records (EFE, posterior, action, …).

    Always writes CSV; additionally writes Parquet when pyarrow is
    available for lossless float round-trip of any vector-valued columns.
    """
    paths.ensure()
    df = pd.DataFrame(records)
    df.to_csv(paths.per_step_csv, index=False)
    try:
        df.to_parquet(paths.per_step_parquet, index=False)
    except (ImportError, ValueError):
        pass
    return paths.per_step_csv


def persist_generative_model_npz(agents: dict, paths: RunPaths) -> Path:
    """Persist each agent's A/B/C/D matrices to a single ``.npz`` archive.

    Keys are namespaced ``<agent_id>/<matrix>`` so a single
    ``np.load(...)`` call exposes every array losslessly (no JSON precision
    drift). Useful when downstream consumers want to instantiate a pymdp
    ``Agent`` from the persisted matrices.
    """
    paths.ensure()
    payload: dict[str, np.ndarray] = {}
    for agent_id, agent in agents.items():
        prefix = f"agent_{agent_id}"
        for name in ("A", "B", "C", "D"):
            arr = getattr(agent, name, None)
            if arr is None:
                continue
            payload[f"{prefix}/{name}"] = np.asarray(arr, dtype=float)
        E = getattr(agent, "E", None)
        if E is not None:
            payload[f"{prefix}/E"] = np.asarray(E)
    np.savez_compressed(paths.matrices_npz, **payload)
    return paths.matrices_npz


def persist_policies(policies: list, paths: RunPaths) -> Path:
    """Persist the enumerated policy set."""
    paths.ensure()
    payload = [_to_jsonable(p) for p in policies]
    with paths.policies_json.open("w", encoding="utf-8") as fh:
        json.dump({"n_policies": len(policies), "policies": payload}, fh, indent=2)
    return paths.policies_json


def _to_jsonable(obj: Any) -> Any:
    """Best-effort conversion to JSON-serialisable primitives."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    except ImportError:  # pragma: no cover
        pass
    return obj
