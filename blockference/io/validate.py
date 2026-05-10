"""Output validation — confirm artefacts on disk are well-formed."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from blockference.io.layout import RunPaths

__all__ = [
    "ValidationReport",
    "validate_generative_model",
    "validate_per_step_records",
    "validate_run_outputs",
    "validate_trajectory_dataframe",
]


REQUIRED_TRAJECTORY_COLUMNS: tuple[str, ...] = (
    "agents",
    "priors",
    "env_states",
    "actions",
    "inferences",
    "efe",
    "efe_epistemic",
    "efe_pragmatic",
    "q_pi",
    "p_u",
    "obs_idx",
    "timestep",
    "substep",
    "run",
)


@dataclass
class ValidationReport:
    """Result of validating a single run's artefacts."""

    ok: bool
    checks: dict[str, bool] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    artefacts: dict[str, str] = field(default_factory=dict)

    def add(self, name: str, ok: bool, issue: str | None = None) -> None:
        self.checks[name] = ok
        if not ok:
            self.ok = False
            if issue:
                self.issues.append(f"{name}: {issue}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checks": self.checks,
            "issues": self.issues,
            "artefacts": self.artefacts,
        }

    def write(self, paths: RunPaths) -> Path:
        paths.ensure()
        with paths.validation_report.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return paths.validation_report


def validate_trajectory_dataframe(df: pd.DataFrame) -> ValidationReport:
    """Lightweight schema check for the cadCAD trajectory frame."""
    report = ValidationReport(ok=True)
    report.add("non_empty", len(df) > 0, f"frame has {len(df)} rows")
    missing = [c for c in REQUIRED_TRAJECTORY_COLUMNS if c not in df.columns]
    report.add(
        "required_columns",
        not missing,
        f"missing columns: {missing}" if missing else None,
    )
    if "env_states" in df.columns:
        sample = df["env_states"].iloc[0]
        report.add(
            "env_states_dict_shape",
            isinstance(sample, dict),
            f"env_states is {type(sample).__name__}, expected dict",
        )
    return report


def validate_run_outputs(paths: RunPaths) -> ValidationReport:
    """End-to-end validation of a run's on-disk artefacts."""
    report = ValidationReport(ok=True)
    report.artefacts = {
        "run_dir": str(paths.run_dir),
        "data_dir": str(paths.data_dir),
        "viz_dir": str(paths.viz_dir),
        "animations_dir": str(paths.animations_dir),
    }
    report.add("run_dir_exists", paths.run_dir.is_dir())
    report.add("data_dir_exists", paths.data_dir.is_dir())
    report.add("viz_dir_exists", paths.viz_dir.is_dir())
    report.add("animations_dir_exists", paths.animations_dir.is_dir())
    report.add("config_persisted", paths.config_path.exists())
    report.add("trajectory_csv_persisted", paths.trajectory_csv.exists())
    report.add("summary_json_persisted", paths.summary_json.exists())
    if paths.trajectory_csv.exists():
        try:
            df = pd.read_csv(paths.trajectory_csv)
            report.add("trajectory_csv_parses", True)
            report.add("trajectory_csv_non_empty", len(df) > 0)
        except Exception as exc:  # pragma: no cover - defensive
            report.add("trajectory_csv_parses", False, str(exc))
    if paths.matrices_json.exists():
        report.add("matrices_persisted", True)
    if paths.matrices_npz.exists():
        report.add("matrices_npz_persisted", True)
    if paths.per_step_csv.exists():
        report.add("per_step_csv_persisted", True)
    if paths.per_step_parquet.exists():
        report.add("per_step_parquet_persisted", True)
    if paths.policies_json.exists():
        report.add("policies_persisted", True)
    if paths.run_log.exists():
        report.add("run_log_persisted", True)
    return report


def validate_per_step_records(records: list[dict[str, Any]]) -> ValidationReport:
    """Validate the math invariants of per-step diagnostics.

    Checks (when the corresponding fields are present):

    * ``posterior`` vectors sum to ~1 and are finite,
    * ``prior`` vectors sum to ~1 and are finite,
    * ``efe`` per-policy is finite,
    * ``q_pi`` and ``p_u`` sum to ~1 and are finite,
    * ``epistemic + pragmatic ≈ efe`` for every (agent, t).
    """
    import numpy as np

    report = ValidationReport(ok=True)
    report.add("non_empty", bool(records), f"{len(records)} records")
    if not records:
        return report

    # Per-vector checks.
    bad_posterior = 0
    bad_prior = 0
    bad_efe = 0
    bad_qpi = 0
    bad_pu = 0
    bad_decomp = 0
    for r in records:
        post = r.get("posterior")
        if isinstance(post, (list, tuple, np.ndarray)):
            a = np.asarray(post, dtype=float)
            if not (np.isfinite(a).all() and np.isclose(a.sum(), 1.0, atol=1e-6)):
                bad_posterior += 1
        prior = r.get("prior")
        if isinstance(prior, (list, tuple, np.ndarray)):
            a = np.asarray(prior, dtype=float)
            if not (np.isfinite(a).all() and np.isclose(a.sum(), 1.0, atol=1e-6)):
                bad_prior += 1
        efe = r.get("efe")
        if isinstance(efe, (list, tuple, np.ndarray)):
            a = np.asarray(efe, dtype=float)
            if not np.isfinite(a).all():
                bad_efe += 1
        qpi = r.get("q_pi")
        if isinstance(qpi, (list, tuple, np.ndarray)):
            a = np.asarray(qpi, dtype=float)
            if not (np.isfinite(a).all() and np.isclose(a.sum(), 1.0, atol=1e-6)):
                bad_qpi += 1
        pu = r.get("p_u")
        if isinstance(pu, (list, tuple, np.ndarray)):
            a = np.asarray(pu, dtype=float)
            if not (np.isfinite(a).all() and np.isclose(a.sum(), 1.0, atol=1e-6)):
                bad_pu += 1
        eps = r.get("efe_epistemic")
        prag = r.get("efe_pragmatic")
        if (
            isinstance(efe, (list, tuple, np.ndarray))
            and isinstance(eps, (list, tuple, np.ndarray))
            and isinstance(prag, (list, tuple, np.ndarray))
        ):
            if not np.allclose(
                np.asarray(efe, dtype=float),
                np.asarray(eps, dtype=float) + np.asarray(prag, dtype=float),
                atol=1e-6,
            ):
                bad_decomp += 1

    report.add("posterior_stochastic", bad_posterior == 0,
               f"{bad_posterior}/{len(records)} bad posterior vectors")
    report.add("prior_stochastic", bad_prior == 0,
               f"{bad_prior}/{len(records)} bad prior vectors")
    report.add("efe_finite", bad_efe == 0,
               f"{bad_efe}/{len(records)} non-finite EFE vectors")
    report.add("q_pi_stochastic", bad_qpi == 0,
               f"{bad_qpi}/{len(records)} bad policy-posterior vectors")
    report.add("p_u_stochastic", bad_pu == 0,
               f"{bad_pu}/{len(records)} bad action-marginal vectors")
    report.add("efe_decomposition_consistent", bad_decomp == 0,
               f"{bad_decomp}/{len(records)} EFE ≠ epistemic+pragmatic")
    return report


def validate_generative_model(agent: Any) -> ValidationReport:
    """Check the math invariants of an :class:`ActiveGridference` agent.

    Confirms (per active Inference theory):

    * ``A`` is a stochastic matrix with the right shape.
    * ``B[:, :, a]`` is column-stochastic for every action ``a``.
    * ``C`` is normalised over observations.
    * ``D`` is normalised over hidden states.
    * Every quantity is finite (no NaN / inf).
    """
    report = ValidationReport(ok=True)
    import numpy as np

    n_states = int(getattr(agent, "n_states", 0))
    n_obs = int(getattr(agent, "n_observations", 0))
    n_actions = len(getattr(agent, "E", []))

    A = getattr(agent, "A", None)
    B = getattr(agent, "B", None)
    C = getattr(agent, "C", None)
    D = getattr(agent, "D", None)

    report.add("A_present", A is not None)
    report.add("B_present", B is not None)
    if A is not None:
        A = np.asarray(A)
        report.add("A_shape", A.shape == (n_obs, n_states),
                   f"expected {(n_obs, n_states)}, got {A.shape}")
        report.add("A_finite", bool(np.isfinite(A).all()))
        col_sums = A.sum(axis=0)
        report.add("A_column_stochastic", bool(np.allclose(col_sums, 1.0, atol=1e-8)),
                   f"col sums min {col_sums.min():.6f} max {col_sums.max():.6f}")
    if B is not None:
        B = np.asarray(B)
        report.add("B_shape", B.shape == (n_states, n_states, n_actions),
                   f"expected {(n_states, n_states, n_actions)}, got {B.shape}")
        report.add("B_finite", bool(np.isfinite(B).all()))
        all_ok = True
        worst = (1.0, 1.0)
        for a in range(B.shape[-1]):
            sums = B[:, :, a].sum(axis=0)
            if not np.allclose(sums, 1.0, atol=1e-8):
                all_ok = False
                if sums.min() < worst[0] or sums.max() > worst[1]:
                    worst = (float(sums.min()), float(sums.max()))
        report.add("B_column_stochastic_per_action", all_ok,
                   f"worst (min, max) = {worst}" if not all_ok else None)
    if C is not None:
        C = np.asarray(C)
        report.add("C_shape", C.shape == (n_obs,),
                   f"expected {(n_obs,)}, got {C.shape}")
        report.add("C_normalised", bool(np.isclose(C.sum(), 1.0, atol=1e-8)),
                   f"sum = {C.sum():.6f}")
    if D is not None:
        D = np.asarray(D)
        report.add("D_shape", D.shape == (n_states,),
                   f"expected {(n_states,)}, got {D.shape}")
        report.add("D_normalised", bool(np.isclose(D.sum(), 1.0, atol=1e-8)),
                   f"sum = {D.sum():.6f}")
    return report
