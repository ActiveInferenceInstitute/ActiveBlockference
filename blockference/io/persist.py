"""Serialisation helpers — write configs, dataframes, and summaries to disk."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from blockference.config import ExperimentConfig
from blockference.io.layout import RunPaths

__all__ = [
    "atomic_replace",
    "persist_config",
    "persist_dataframe",
    "persist_generative_model",
    "persist_generative_model_npz",
    "persist_manifest",
    "persist_per_step_records",
    "persist_policies",
    "persist_summary",
]


def atomic_replace(path: Path, writer: Callable[[Path], None]) -> Path:
    """Write ``path`` through a sibling temporary file and publish atomically.

    The payload is written to a unique temporary file in the same directory and
    then ``os.replace``-d into place, so a reader or a crashed process only ever
    observes either the previous complete file or the new complete file — never
    a partially written one. The temporary file is removed on failure.
    """
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=directory
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        writer(temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _atomic_json(path: Path, payload: Any) -> Path:
    """Write JSON through a sibling temporary file and publish it atomically."""

    def _write(target: Path) -> None:
        with target.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")

    return atomic_replace(path, _write)


def persist_config(cfg: ExperimentConfig, paths: RunPaths) -> Path:
    """Write the run's config back to disk so the artefacts are self-describing."""
    paths.require_tree()

    def _write(target: Path) -> None:
        with target.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(_to_jsonable(cfg.to_dict()), fh, sort_keys=False)

    return atomic_replace(paths.config_path, _write)


def persist_dataframe(
    df: pd.DataFrame, paths: RunPaths, *, models: dict[tuple[int, str], Any]
) -> dict[str, Path]:
    """Write a trajectory whose agent references bind each run's immutable model."""
    paths.require_tree()
    written: dict[str, Path] = {}
    serialised = df.copy()
    references = {
        key: {"agent_id": key[1], "model_sha256": _model_digest(_model_payload(agent))}
        for key, agent in models.items()
    }
    for column in ("agents", "priors", "env_states", "actions", "inferences", "efe", "efe_epistemic", "efe_pragmatic", "q_pi", "p_u", "obs_idx"):
        if column not in serialised:
            continue
        if column == "agents":
            serialised[column] = [
                {
                    str(agent_id): references[(int(run), str(agent_id))]
                    for agent_id in agents
                }
                for run, agents in zip(df["run"], df["agents"], strict=True)
            ]
        else:
            serialised[column] = serialised[column].map(_to_jsonable)
    atomic_replace(paths.trajectory_csv, lambda target: serialised.to_csv(target, index=False))
    written["csv"] = paths.trajectory_csv
    try:
        atomic_replace(
            paths.trajectory_parquet, lambda target: serialised.to_parquet(target, index=False)
        )
        written["parquet"] = paths.trajectory_parquet
    except ImportError:
        # Parquet is an explicitly optional acceleration format.
        pass
    return written


def persist_summary(summary: dict[str, Any], paths: RunPaths) -> Path:
    """Write the per-run summary JSON (number of agents, final positions, ...)."""
    paths.require_tree()
    return _atomic_json(paths.summary_json, _to_jsonable(summary))


def _model_payload(agent: Any) -> dict[str, Any]:
    return {
        "n_states": int(getattr(agent, "n_states", 0)),
        "n_observations": int(getattr(agent, "n_observations", 0)),
        "border": int(getattr(agent, "border", 0)),
        "policy_len": int(getattr(agent, "policy_len", 0)),
        "max_policies": int(getattr(agent, "max_policies", 100_000)),
        "E": list(getattr(agent, "E", [])),
        **{name: _to_jsonable(getattr(agent, name, None)) for name in ("A", "B", "C", "D")},
    }


def _model_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _run_payload(values: dict[tuple[int, str], Any]) -> dict[str, Any]:
    runs: dict[str, dict[str, Any]] = {}
    for (run, agent_id), value in values.items():
        if type(run) is not int or run < 1 or not isinstance(agent_id, str) or not agent_id:
            raise ValueError("model keys must be (positive run, non-empty agent ID string)")
        runs.setdefault(str(run), {})[agent_id] = value
    return {"version": 2, "runs": runs}


def persist_generative_model(agents: dict[tuple[int, str], Any], paths: RunPaths) -> Path:
    """Persist immutable A/B/C/D/E and planning metadata by ``(run, agent)``."""
    paths.require_tree()
    if not agents:
        raise ValueError("at least one agent is required to persist a generative model")
    payload = {key: _model_payload(agent) for key, agent in agents.items()}
    return _atomic_json(paths.matrices_json, _run_payload(payload))


def persist_per_step_records(records: list[dict[str, Any]], paths: RunPaths) -> Path:
    """Write per-step / per-agent records (EFE, posterior, action, …).

    Always writes CSV; additionally writes Parquet when pyarrow is
    available for lossless float round-trip of any vector-valued columns.
    """
    paths.require_tree()
    if not records:
        raise ValueError("at least one per-step record is required")
    df = pd.DataFrame(records)
    atomic_replace(paths.per_step_csv, lambda target: df.to_csv(target, index=False))
    try:
        atomic_replace(paths.per_step_parquet, lambda target: df.to_parquet(target, index=False))
    except ImportError:
        # Parquet is an explicitly optional acceleration format.
        pass
    return paths.per_step_csv


def persist_generative_model_npz(agents: dict[tuple[int, str], Any], paths: RunPaths) -> Path:
    """Persist matrices with ``run_<run>/agent_<agent>/<matrix>`` archive keys."""
    paths.require_tree()
    if not agents:
        raise ValueError("at least one agent is required to persist matrices")
    payload: dict[str, np.ndarray] = {}
    for (run, agent_id), agent in agents.items():
        prefix = f"run_{run}/agent_{agent_id}"
        for name in ("A", "B", "C", "D"):
            arr = getattr(agent, name, None)
            if arr is None:
                continue
            payload[f"{prefix}/{name}"] = np.asarray(arr, dtype=float)
        E = getattr(agent, "E", None)
        if E is not None:
            payload[f"{prefix}/E"] = np.asarray(E)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **payload)  # type: ignore[arg-type]
    bytes_payload = buffer.getvalue()

    def _write(target: Path) -> None:
        target.write_bytes(bytes_payload)

    return atomic_replace(paths.matrices_npz, _write)


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

    def _write(target: Path) -> None:
        with target.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")

    return atomic_replace(paths.manifest_json, _write)


def persist_policies(policies: dict[tuple[int, str], list], paths: RunPaths) -> Path:
    """Persist each run/agent's actual allowed, ordered policy set."""
    paths.require_tree()
    if not policies or any(not values for values in policies.values()):
        raise ValueError("at least one policy per run/agent is required")
    payload = {
        key: {"n_policies": len(values), "policies": [_to_jsonable(p) for p in values]}
        for key, values in policies.items()
    }
    return _atomic_json(paths.policies_json, _run_payload(payload))


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
