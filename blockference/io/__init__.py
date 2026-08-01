"""Output organisation, persistence, and validation helpers."""

from blockference.io.layout import RunPaths, build_run_paths
from blockference.io.persist import (
    atomic_replace,
    persist_config,
    persist_dataframe,
    persist_generative_model,
    persist_generative_model_npz,
    persist_manifest,
    persist_per_step_records,
    persist_policies,
    persist_summary,
)
from blockference.io.schema import TrajectoryRecord, parse_trajectory_records
from blockference.io.validate import (
    ValidationReport,
    validate_generative_model,
    validate_per_step_records,
    validate_run_outputs,
    validate_trajectory_dataframe,
)

__all__ = [
    "RunPaths",
    "TrajectoryRecord",
    "ValidationReport",
    "atomic_replace",
    "build_run_paths",
    "parse_trajectory_records",
    "persist_config",
    "persist_dataframe",
    "persist_generative_model",
    "persist_generative_model_npz",
    "persist_manifest",
    "persist_per_step_records",
    "persist_policies",
    "persist_summary",
    "validate_generative_model",
    "validate_per_step_records",
    "validate_run_outputs",
    "validate_trajectory_dataframe",
]
