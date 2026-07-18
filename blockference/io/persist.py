"""Serialisation helpers — write configs, dataframes, and summaries to disk."""

from __future__ import annotations

import hashlib
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
    "persist_manifest",
    "persist_per_step_records",
    "persist_policies",
    "persist_summary",
]


def _atomic_json(path: Path, payload: Any) -> Path:
    """Write JSON through a sibling temporary file and publish it atomically."""

    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    temporary.replace(path)
    return path


def persist_config(cfg: ExperimentConfig, paths: RunPaths) -> Path:
    """Write the run's config back to disk so the artefacts are self-describing."""
    paths.require_tree()
    temporary = paths.config_path.with_name(f".{paths.config_path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(_to_jsonable(cfg.to_dict()), fh, sort_keys=False)
    temporary.replace(paths.config_path)
    return paths.config_path


def persist_dataframe(df: pd.DataFrame, paths: RunPaths) -> dict[str, Path]:
    """Write a documented nested-value trajectory as CSV and optional Parquet."""
    paths.require_tree()
    written: dict[str, Path] = {}
    serialised = df.copy()
    for column in ("agents", "priors", "env_states", "actions", "inferences", "efe", "efe_epistemic", "efe_pragmatic", "q_pi", "p_u", "obs_idx"):
        if column not in serialised:
            continue
        if column == "agents":
            serialised[column] = serialised[column].map(
                lambda value: {str(key): {"agent_id": str(key)} for key in value}
                if isinstance(value, dict)
                else value
            )
        else:
            serialised[column] = serialised[column].map(_to_jsonable)
    serialised.to_csv(paths.trajectory_csv, index=False)
    written["csv"] = paths.trajectory_csv
    try:
        serialised.to_parquet(paths.trajectory_parquet, index=False)
        written["parquet"] = paths.trajectory_parquet
    except ImportError:
        # Parquet is an explicitly optional acceleration format.
        pass
    return written


def persist_summary(summary: dict[str, Any], paths: RunPaths) -> Path:
    """Write the per-run summary JSON (number of agents, final positions, ...)."""
    paths.require_tree()
    return _atomic_json(paths.summary_json, _to_jsonable(summary))


def persist_generative_model(agents: dict, paths: RunPaths) -> Path:
    """Persist each agent's A/B/C/D/E matrices to a single JSON file.

    Parameters
    ----------
    agents :
        ``{agent_id: ActiveGridference}`` mapping. Only the public matrices
        are serialised; per-step state is captured in :func:`persist_per_step_records`.
    """
    paths.require_tree()
    if not agents:
        raise ValueError("at least one agent is required to persist a generative model")
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
    return _atomic_json(paths.matrices_json, payload)


def persist_per_step_records(records: list[dict[str, Any]], paths: RunPaths) -> Path:
    """Write per-step / per-agent records (EFE, posterior, action, …).

    Always writes CSV; additionally writes Parquet when pyarrow is
    available for lossless float round-trip of any vector-valued columns.
    """
    paths.require_tree()
    if not records:
        raise ValueError("at least one per-step record is required")
    df = pd.DataFrame(records)
    df.to_csv(paths.per_step_csv, index=False)
    try:
        df.to_parquet(paths.per_step_parquet, index=False)
    except ImportError:
        # Parquet is an explicitly optional acceleration format.
        pass
    return paths.per_step_csv


def persist_generative_model_npz(agents: dict, paths: RunPaths) -> Path:
    """Persist each agent's A/B/C/D matrices to a single ``.npz`` archive.

    Keys are namespaced ``<agent_id>/<matrix>`` so a single
    ``np.load(...)`` call exposes every array losslessly (no JSON precision
    drift). Useful when downstream consumers want to instantiate a pymdp
    ``Agent`` from the persisted matrices.
    """
    paths.require_tree()
    if not agents:
        raise ValueError("at least one agent is required to persist matrices")
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
    np.savez_compressed(paths.matrices_npz, **payload)  # type: ignore[arg-type]
    return paths.matrices_npz


def persist_manifest(paths: RunPaths) -> Path:
    """Write hashes and sizes for every stable artifact in a run tree.

    The manifest intentionally excludes itself, the mutable validation report,
    and the timestamped run log. Those files have their own presence and
    parseability checks and must not create a circular or perpetually changing
    digest.
    """

    paths.require_tree()
    excluded = {paths.manifest_json, paths.validation_report, paths.run_log}
    files: list[dict[str, Any]] = []
    for path in sorted(paths.run_dir.rglob("*")):
        if not path.is_file() or path in excluded:
            continue
        relative = path.relative_to(paths.run_dir).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append({"path": relative, "sha256": digest, "size_bytes": path.stat().st_size})
    payload = {"version": 1, "complete": True, "files": files}
    temporary = paths.manifest_json.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(paths.manifest_json)
    return paths.manifest_json


def persist_policies(policies: list, paths: RunPaths) -> Path:
    """Persist the enumerated policy set."""
    paths.require_tree()
    if not policies:
        raise ValueError("at least one policy is required")
    payload = [_to_jsonable(p) for p in policies]
    return _atomic_json(
        paths.policies_json,
        {"n_policies": len(policies), "policies": payload},
    )


def _to_jsonable(obj: Any) -> Any:
    """Best-effort conversion to JSON-serialisable primitives."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj
