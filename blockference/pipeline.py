"""End-to-end run pipeline for ActiveBlockference experiments.

Composable stages — each stage is a small pure(ish) function so that
callers can mix-and-match (e.g. run only the viz stage on an existing
trajectory CSV, only the validation stage in CI, or replace the simulate
stage with a custom radCAD/cadCAD harness).

Stages
------

* :func:`simulate`               run radCAD/cadCAD and return the result frame
* :func:`persist`                write trajectory, summary, generative model
                                 (JSON + NPZ), policies, per-step records
* :func:`render_visualisations`  produce PNGs + GIF
* :func:`validate`               run schema + math + artefact validation

The convenience entry point :func:`run_pipeline` glues all four stages
together; tests / notebooks may call any subset directly.

Artefacts emitted
-----------------

* ``config.yml``                          — the config that produced this run
* ``data/trajectory.csv``                 — radCAD/cadCAD result frame
* ``data/trajectory.parquet``             — same, parquet (when pyarrow available)
* ``data/summary.json``                   — n_agents, final positions, …
* ``data/generative_model.json``          — A/B/C/D/E for every agent (JSON)
* ``data/generative_model.npz``           — A/B/C/D/E for every agent (binary)
* ``data/per_step.csv`` / ``.parquet``    — per-(agent, t) EFE / qs / Q_pi / P_u / action
* ``data/policies.json``                  — enumerated policy set
* ``viz/*.png`` and ``animations/trajectory.gif``
* ``run.log``                             — structured logs from this run
* ``validation_report.json``              — schema + math + artefact validation
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from blockference.config import ExperimentConfig, load_experiment_config
from blockference.io import (
    RunPaths,
    ValidationReport,
    build_run_paths,
    persist_config,
    persist_dataframe,
    persist_generative_model,
    persist_generative_model_npz,
    persist_per_step_records,
    persist_policies,
    persist_summary,
    validate_generative_model,
    validate_per_step_records,
    validate_run_outputs,
    validate_trajectory_dataframe,
)
from blockference.logging_setup import configure_run_logging, get_logger, remove_handler
from blockference.viz import (
    animate_trajectory,
    plot_action_distribution,
    plot_belief_heatmap,
    plot_efe_proxy,
    plot_trajectory,
)
from blockference.viz._common import (
    extract_agent_positions,
    extract_belief_history,
    infer_grid_dimension,
)

__all__ = [
    "PipelineResult",
    "build_per_step_records",
    "persist",
    "render_visualisations",
    "run_pipeline",
    "simulate",
    "summarise_trajectory",
    "validate",
]


log = get_logger(__name__)


@dataclass
class PipelineResult:
    """Bundle of artefacts produced by :func:`run_pipeline`."""

    config: ExperimentConfig
    paths: RunPaths
    trajectory: pd.DataFrame
    summary: dict
    schema_report: ValidationReport
    artefacts_report: ValidationReport
    model_reports: dict[str, ValidationReport]
    per_step_report: ValidationReport
    visualisations: dict[str, Path]
    animations: dict[str, Path]


def summarise_trajectory(df: pd.DataFrame) -> dict:
    """Compute a JSON-friendly summary of an experiment run."""
    positions = extract_agent_positions(df)
    beliefs = extract_belief_history(df)
    n = infer_grid_dimension(df)
    summary = {
        "n_rows": int(len(df)),
        "n_agents": int(len(positions)),
        "grid_dimension": int(n),
        "final_positions": {str(k): v[-1] if v else None for k, v in positions.items()},
        "trajectory_lengths": {str(k): len(v) for k, v in positions.items()},
        "belief_history_lengths": {str(k): len(v) for k, v in beliefs.items()},
    }
    if "timestep" in df.columns:
        summary["max_timestep"] = int(df["timestep"].max())
    if "run" in df.columns:
        summary["max_run"] = int(df["run"].max())
    return summary


def _as_float_array(v: Any) -> np.ndarray | None:
    """Coerce a cell that may be list/tuple/ndarray/str-of-list into ``np.ndarray``."""
    if isinstance(v, np.ndarray):
        return v.astype(float, copy=False)
    if isinstance(v, (list, tuple)):
        return np.asarray(v, dtype=float)
    return None


def _entropy(p: np.ndarray) -> float:
    return float(-(p * np.log(np.clip(p, 1e-16, 1.0))).sum())


def build_per_step_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Flatten the cadCAD frame to one record per (agent, run, substep, timestep).

    Captures the **full** per-step diagnostic surface: ``q(s)`` posterior,
    prior, sampled action, observation index, per-policy ``G(π)`` with
    epistemic + pragmatic decomposition, policy posterior ``Q(π)``,
    action marginal ``P(u)``. Vector-valued columns are stored as Python
    lists so that the CSV form survives a round-trip via ``ast.literal_eval``;
    the parallel parquet form keeps them as nested arrays losslessly.
    """
    records: list[dict[str, Any]] = []
    if df.empty:
        return records
    for _, row in df.iterrows():
        env = row.get("env_states", {}) or {}
        actions = row.get("actions", {}) or {}
        inferences = row.get("inferences", {}) or {}
        priors = row.get("priors", {}) or {}
        efes = row.get("efe", {}) or {}
        efes_eps = row.get("efe_epistemic", {}) or {}
        efes_prag = row.get("efe_pragmatic", {}) or {}
        q_pis = row.get("q_pi", {}) or {}
        p_us = row.get("p_u", {}) or {}
        obs_idxs = row.get("obs_idx", {}) or {}
        for agent_id in env:
            qs_arr = _as_float_array(inferences.get(agent_id))
            prior_arr = _as_float_array(priors.get(agent_id))
            efe_arr = _as_float_array(efes.get(agent_id))
            eps_arr = _as_float_array(efes_eps.get(agent_id))
            prag_arr = _as_float_array(efes_prag.get(agent_id))
            q_pi_arr = _as_float_array(q_pis.get(agent_id))
            p_u_arr = _as_float_array(p_us.get(agent_id))

            rec: dict[str, Any] = {
                "run": row.get("run"),
                "substep": row.get("substep"),
                "timestep": row.get("timestep"),
                "agent_id": agent_id,
                "env_state": env.get(agent_id),
                "action": actions.get(agent_id),
                "obs_idx": obs_idxs.get(agent_id),
                "posterior": qs_arr.tolist() if qs_arr is not None else None,
                "prior": prior_arr.tolist() if prior_arr is not None else None,
                "efe": efe_arr.tolist() if efe_arr is not None else None,
                "efe_epistemic": eps_arr.tolist() if eps_arr is not None else None,
                "efe_pragmatic": prag_arr.tolist() if prag_arr is not None else None,
                "q_pi": q_pi_arr.tolist() if q_pi_arr is not None else None,
                "p_u": p_u_arr.tolist() if p_u_arr is not None else None,
                "posterior_entropy": _entropy(qs_arr) if qs_arr is not None else None,
                "posterior_max_prob": float(qs_arr.max()) if qs_arr is not None else None,
                "prior_entropy": _entropy(prior_arr) if prior_arr is not None else None,
                "efe_min": float(efe_arr.min()) if efe_arr is not None else None,
                "efe_argmin": int(efe_arr.argmin()) if efe_arr is not None else None,
                "q_pi_entropy": _entropy(q_pi_arr) if q_pi_arr is not None else None,
                "p_u_entropy": _entropy(p_u_arr) if p_u_arr is not None else None,
            }
            records.append(rec)
    return records


def render_visualisations(
    df: pd.DataFrame,
    paths: RunPaths,
    *,
    affordances: list[str] | None = None,
) -> tuple[dict[str, Path], dict[str, Path]]:
    """Render the standard PNG + GIF artefacts. Returns (viz, animations) maps."""
    paths.ensure()
    viz: dict[str, Path] = {}
    anims: dict[str, Path] = {}

    log.info("rendering trajectory plot")
    viz["trajectory"] = plot_trajectory(df, paths.viz_dir / "trajectory.png")
    log.info("rendering action distribution")
    viz["actions"] = plot_action_distribution(
        df, paths.viz_dir / "action_distribution.png", affordances=affordances
    )
    log.info("rendering EFE proxy (belief entropy)")
    viz["efe_proxy"] = plot_efe_proxy(df, paths.viz_dir / "efe_proxy.png")

    beliefs = extract_belief_history(df)
    if beliefs:
        first_agent = next(iter(beliefs))
        n_steps = len(beliefs[first_agent])
        if n_steps > 0:
            log.info("rendering belief heatmaps for agent %s (first/last)", first_agent)
            viz["belief_heatmap_first"] = plot_belief_heatmap(
                df,
                paths.viz_dir / "belief_heatmap_first.png",
                agent_id=first_agent,
                timestep=0,
            )
            viz["belief_heatmap_last"] = plot_belief_heatmap(
                df,
                paths.viz_dir / "belief_heatmap_last.png",
                agent_id=first_agent,
                timestep=-1,
            )

    try:
        log.info("rendering trajectory animation")
        anims["trajectory"] = animate_trajectory(df, paths.animations_dir / "trajectory.gif")
    except (ValueError, RuntimeError) as exc:
        log.warning("animation skipped: %s", exc)
    return viz, anims


def _extract_agents_dict(df: pd.DataFrame) -> dict:
    """Return the post-run ``agents`` dict from the final cadCAD row."""
    if df.empty or "agents" not in df.columns:
        return {}
    last_agents = df.iloc[-1]["agents"]
    if isinstance(last_agents, dict):
        return last_agents
    return {}


def _extract_policies(agents: dict) -> list:
    """Return the policy set the inference loop used."""
    from blockference.maths import construct_policies as _cp

    if not agents:
        return []
    first = next(iter(agents.values()))
    n_states = int(getattr(first, "n_states", 0))
    n_actions = len(getattr(first, "E", []))
    policy_len = int(getattr(first, "policy_len", 1))
    if n_states == 0 or n_actions == 0:
        return []
    return _cp([n_states], [n_actions], policy_len=policy_len)


# ---------------------------------------------------------------------------
# Composable stages
# ---------------------------------------------------------------------------


def simulate(cfg: ExperimentConfig) -> pd.DataFrame:
    """Stage 1: run the radCAD/cadCAD experiment and return the result frame."""
    from blockference.simulations import run_experiment

    log.info(
        "simulate: engine=%s timesteps=%d runs=%d",
        cfg.engine,
        cfg.simulation.timesteps,
        cfg.simulation.runs,
    )
    df = run_experiment(cfg)
    log.info("simulate: produced %d rows", len(df))
    return df


def persist(
    cfg: ExperimentConfig,
    df: pd.DataFrame,
    paths: RunPaths,
) -> tuple[dict, list[dict[str, Any]], dict]:
    """Stage 2: write every artefact to ``paths``.

    Returns ``(summary, per_step_records, agents_dict)`` so downstream
    stages can reuse the in-memory state without re-reading the CSV.
    """
    paths.ensure()
    persist_config(cfg, paths)
    log.info("persisted config to %s", paths.config_path)

    written = persist_dataframe(df, paths)
    log.info("persisted trajectory: %s", {k: str(v) for k, v in written.items()})

    summary = summarise_trajectory(df)
    persist_summary(summary, paths)
    log.info("persisted summary: %d agents, %d rows", summary["n_agents"], summary["n_rows"])

    agents = _extract_agents_dict(df)
    if agents:
        persist_generative_model(agents, paths)
        persist_generative_model_npz(agents, paths)
        log.info("persisted generative model (JSON + NPZ) for %d agent(s)", len(agents))
        policies = _extract_policies(agents)
        if policies:
            persist_policies(policies, paths)
            log.info("persisted %d policies of length %d", len(policies), policies[0].shape[0])

    per_step = build_per_step_records(df)
    if per_step:
        persist_per_step_records(per_step, paths)
        log.info("persisted %d per-step records", len(per_step))
    return summary, per_step, agents


def validate(
    df: pd.DataFrame,
    agents: dict,
    per_step: list[dict[str, Any]],
    paths: RunPaths,
) -> tuple[ValidationReport, dict[str, ValidationReport], ValidationReport, ValidationReport]:
    """Stage 3: schema + math + per-step + on-disk validation.

    Writes ``validation_report.json`` covering all on-disk artefacts and
    returns the four sub-reports so callers can inspect each axis.
    """
    schema_report = validate_trajectory_dataframe(df)
    if not schema_report.ok:
        log.warning("trajectory schema issues: %s", schema_report.issues)
    else:
        log.info("trajectory schema OK")

    model_reports: dict[str, ValidationReport] = {}
    for agent_id, agent in agents.items():
        rep = validate_generative_model(agent)
        model_reports[str(agent_id)] = rep
        if not rep.ok:
            log.warning(
                "generative model invariant failures for agent %s: %s", agent_id, rep.issues
            )
        else:
            log.info("generative model invariants OK for agent %s", agent_id)

    per_step_report = validate_per_step_records(per_step)
    if not per_step_report.ok:
        log.warning("per-step diagnostics issues: %s", per_step_report.issues)
    else:
        log.info("per-step diagnostics OK")

    artefacts_report = validate_run_outputs(paths)
    artefacts_report.write(paths)
    if not artefacts_report.ok:
        log.warning("artefact validation issues: %s", artefacts_report.issues)
    else:
        log.info("artefact validation OK")
    return schema_report, model_reports, per_step_report, artefacts_report


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def run_pipeline(
    cfg: ExperimentConfig | str | Path,
    *,
    output_root: str | Path = "output",
    run_name: str | None = None,
    timestamped: bool = False,
) -> PipelineResult:
    """Run an experiment end-to-end and emit a fully populated ``output/`` tree."""
    if not isinstance(cfg, ExperimentConfig):
        cfg = load_experiment_config(cfg)

    paths = build_run_paths(
        root=output_root,
        run_name=run_name or cfg.name,
        timestamped=timestamped,
    )

    log_handler = configure_run_logging(paths.run_log)
    try:
        log.info("=== ActiveBlockference run start: %s ===", paths.run_name)
        log.info("config: name=%s seed=%s engine=%s", cfg.name, cfg.seed, cfg.engine)
        log.info(
            "grid: dim=%d planning_length=%d affordances=%s",
            cfg.grid.dimension,
            cfg.grid.planning_length,
            cfg.grid.affordances,
        )
        log.info(
            "simulation: timesteps=%d runs=%d n_agents=%d target=%s initial_state=%s",
            cfg.simulation.timesteps,
            cfg.simulation.runs,
            cfg.simulation.n_agents,
            cfg.simulation.target,
            cfg.simulation.initial_state,
        )

        # Route the simulation's CSV writer through our typed path.
        cfg.output.path = str(paths.trajectory_csv)

        df = simulate(cfg)
        summary, per_step, agents = persist(cfg, df, paths)

        # Visualisations after persistence so the run_log captures both phases.
        visualisations, animations = render_visualisations(
            df, paths, affordances=cfg.grid.affordances
        )

        schema_report, model_reports, per_step_report, artefacts_report = validate(
            df, agents, per_step, paths
        )

        log.info("=== ActiveBlockference run complete: %s ===", paths.run_name)

        return PipelineResult(
            config=cfg,
            paths=paths,
            trajectory=df,
            summary=summary,
            schema_report=schema_report,
            artefacts_report=artefacts_report,
            model_reports=model_reports,
            per_step_report=per_step_report,
            visualisations=visualisations,
            animations=animations,
        )
    finally:
        remove_handler(log_handler)
