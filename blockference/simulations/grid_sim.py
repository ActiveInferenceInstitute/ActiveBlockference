"""radCAD/cadCAD-driven multi-agent grid simulation entry point.

Two ways to invoke:

1. Programmatic — pass primitives or an ``ExperimentConfig``::

       from blockference.simulations import run_grid
       df = run_grid(dimension=3, no_agents=2, no_timesteps=10)

       from blockference.config import load_experiment_config
       from blockference.simulations import run_experiment
       df = run_experiment(load_experiment_config("configs/example.yml"))

2. CLI::

       python -m blockference.simulations.grid_sim configs/example.yml
       python -m blockference.simulations.grid_sim 3 2 10        # legacy form

The CLI auto-detects whether the first arg is a path or an integer.

Engine selection
----------------
``cfg.engine == "radcad"`` (default) drives ``radcad.Model`` /
``radcad.Simulation`` — the fast, parallelisable cadCAD successor.
``cfg.engine == "cadcad"`` routes the same state-update blocks through
upstream ``cadCAD`` (slower, but the canonical reference implementation
shipped by BlockScience). Both backends consume the same PSUBs so
behaviour is bit-identical up to RNG draws.
"""

from __future__ import annotations

import argparse
import itertools
import os
import random as rand
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from blockference.config import ExperimentConfig, load_experiment_config
from blockference.gridference import ActiveGridference
from blockference.utils.policy import p_actinf_dict

__all__ = ["run_grid", "run_experiment", "build_state_update_blocks", "build_initial_state"]


# Per-agent diagnostic fields surfaced into cadCAD state at every step.
# Kept here so the persistence + validation modules can introspect the
# canonical field list without importing anything radcad-specific.
PER_STEP_FIELDS: tuple[tuple[str, str], ...] = (
    ("priors", "update_prior"),
    ("env_states", "update_env"),
    ("actions", "update_action"),
    ("inferences", "update_inference"),
    ("efe", "update_efe"),
    ("efe_epistemic", "update_efe_epistemic"),
    ("efe_pragmatic", "update_efe_pragmatic"),
    ("q_pi", "update_q_pi"),
    ("p_u", "update_p_u"),
    ("obs_idx", "update_obs_idx"),
)


def build_initial_state(grid, n_agents, dimension, target, initial_state, planning_length):
    """Construct the initial cadCAD state dict for a multi-agent gridworld."""
    state = {
        "agents": {},
        **{key: {} for key, _ in PER_STEP_FIELDS},
    }
    for a in range(int(n_agents)):
        agent = ActiveGridference(grid, planning_length=planning_length)
        if target == "random":
            t = (rand.randint(0, int(dimension) - 1), rand.randint(0, int(dimension) - 1))
        else:
            t = tuple(target)
        agent.get_C(t)
        agent.get_D(tuple(initial_state))
        state["agents"][a] = agent
        state["priors"][a] = agent.D
        state["env_states"][a] = agent.env_state
        state["inferences"][a] = agent.current_inference
        state["actions"][a] = agent.current_action
        # Diagnostic slots start empty; populated by the policy block.
        state["efe"][a] = ""
        state["efe_epistemic"][a] = ""
        state["efe_pragmatic"][a] = ""
        state["q_pi"][a] = ""
        state["p_u"][a] = ""
        state["obs_idx"][a] = ""
    return state


# Back-compat alias for any caller that imported the historical private name.
_build_initial_state = build_initial_state


def _make_state_updater(state_key: str, update_key: str):
    """Return a cadCAD state-update fn that copies ``state_key`` and applies updates."""

    def s_fn(params, substep, state_history, previous_state, policy_input):
        new = previous_state[state_key].copy()
        for upd in policy_input.get("agent_updates", []):
            new[upd["source"]] = upd[update_key]
        return state_key, new

    s_fn.__name__ = f"s_{state_key}"
    return s_fn


def _agents_state_update(params, substep, state_history, previous_state, policy_input):
    agents_new = previous_state["agents"].copy()
    for upd in policy_input.get("agent_updates", []):
        agent = agents_new[upd["source"]]
        agent.prior = upd["update_prior"]
        agent.env_state = upd["update_env"]
        agent.current_action = upd["update_action"]
        agent.current_inference = upd["update_inference"]
    return "agents", agents_new


def build_state_update_blocks(grid):
    """Assemble the canonical multi-agent PSUB list.

    Exposed as a public helper so callers driving cadCAD/radCAD by hand
    (or composing additional updaters) can reuse the same wiring the
    pipeline relies on.
    """
    def policy(params, substep, state_history, previous_state):
        return p_actinf_dict(params, substep, state_history, previous_state, grid)

    variables = {"agents": _agents_state_update}
    for key, upd in PER_STEP_FIELDS:
        variables[key] = _make_state_updater(key, upd)

    return [{"policies": {"p_actinf": policy}, "variables": variables}]


def _run_radcad(initial_state, state_update_blocks, params, timesteps, runs):
    from radcad import Model, Simulation

    model = Model(
        initial_state=initial_state,
        state_update_blocks=state_update_blocks,
        params=params,
    )
    simulation = Simulation(model=model, timesteps=int(timesteps), runs=int(runs))
    return simulation.run()


def _run_cadcad(initial_state, state_update_blocks, params, timesteps, runs):
    """Drive the same PSUBs through upstream cadCAD."""
    from cadCAD.configuration import Experiment
    from cadCAD.configuration.utils import config_sim
    from cadCAD.engine import ExecutionContext, ExecutionMode, Executor

    sim_config = config_sim({
        "N": int(runs),
        "T": range(int(timesteps)),
        "M": params,
    })

    experiment = Experiment()
    experiment.append_configs(
        initial_state=initial_state,
        partial_state_update_blocks=state_update_blocks,
        sim_configs=sim_config,
    )
    exec_context = ExecutionContext(context=ExecutionMode().local_mode)
    executor = Executor(exec_context=exec_context, configs=experiment.configs)
    raw_result, _tensor_field, _sessions = executor.execute()
    return raw_result


def run_experiment(cfg: ExperimentConfig) -> pd.DataFrame:
    """Run a configured cadCAD/radCAD multi-agent gridworld and return the result frame.

    Honours ``cfg.seed`` (Python ``random`` + NumPy global RNG), writes
    ``cfg.output.path`` if set, and selects the engine via ``cfg.engine``.
    """
    if cfg.seed is not None:
        rand.seed(cfg.seed)
        np.random.seed(cfg.seed)

    grid = list(itertools.product(range(int(cfg.grid.dimension)), repeat=2))
    initial_state = build_initial_state(
        grid=grid,
        n_agents=cfg.simulation.n_agents,
        dimension=cfg.grid.dimension,
        target=cfg.simulation.target,
        initial_state=cfg.simulation.initial_state,
        planning_length=cfg.grid.planning_length,
    )

    params = {
        "preferred_state": [grid],
        "initial_state": [grid],
        "noise": [0],
    }

    state_update_blocks = build_state_update_blocks(grid)

    engine = (getattr(cfg, "engine", None) or "radcad").lower()
    if engine == "radcad":
        result = _run_radcad(
            initial_state, state_update_blocks, params,
            cfg.simulation.timesteps, cfg.simulation.runs,
        )
    elif engine == "cadcad":
        result = _run_cadcad(
            initial_state, state_update_blocks, params,
            cfg.simulation.timesteps, cfg.simulation.runs,
        )
    else:
        raise ValueError(f"unknown engine {engine!r}; use 'radcad' or 'cadcad'")

    df = pd.DataFrame(result)

    if cfg.output.path:
        out = Path(cfg.output.path)
        if out.parent and str(out.parent) not in ("", "."):
            out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out)
    return df


def run_grid(
    dimension,
    no_agents,
    no_timesteps,
    output_path: str | None = "result.csv",
    *,
    planning_length: int = 2,
    target=("random",),
    initial_state=(0, 0),
    seed: int | None = None,
    runs: int = 1,
    engine: str = "radcad",
) -> pd.DataFrame:
    """Convenience wrapper that builds an :class:`ExperimentConfig` from primitives.

    Mirrors the historical signature for backwards compatibility while
    routing through :func:`run_experiment`.
    """
    target_value = "random"
    if target and target != ("random",):
        target_value = target  # type: ignore[assignment]

    cfg = ExperimentConfig.from_dict({
        "name": "run_grid",
        "seed": seed,
        "engine": engine,
        "grid": {
            "dimension": int(dimension),
            "planning_length": planning_length,
        },
        "simulation": {
            "timesteps": int(no_timesteps),
            "runs": runs,
            "n_agents": int(no_agents),
            "target": target_value,
            "initial_state": list(initial_state),
        },
        "output": {"path": output_path},
    })
    print(
        f"Running Gridference. Grid dimension: {cfg.grid.dimension} | "
        f"Number of agents: {cfg.simulation.n_agents} | "
        f"Timesteps: {cfg.simulation.timesteps} | "
        f"Engine: {cfg.engine}"
    )
    return run_experiment(cfg)


def _looks_like_path(s: str) -> bool:
    return s.endswith((".yml", ".yaml", ".toml")) or os.sep in s or "/" in s


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run an Active Inference cadCAD/radCAD gridworld experiment."
    )
    parser.add_argument(
        "args",
        nargs="*",
        help=(
            "Either a path to a YAML/TOML config, or three integers: "
            "<dimension> <n_agents> <timesteps>."
        ),
    )
    parser.add_argument(
        "--config", "-c", help="Path to a YAML/TOML experiment config (overrides positional)."
    )
    parser.add_argument("--output", "-o", help="Override output CSV path.")
    parser.add_argument(
        "--engine",
        choices=["radcad", "cadcad"],
        help="Engine override (defaults to cfg.engine or 'radcad').",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help=(
            "Run the full pipeline (data + viz + animation + validation) "
            "into output/<run_name>/ instead of writing only a CSV."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="output",
        help="Root directory for pipeline outputs (default: output/).",
    )
    parser.add_argument(
        "--run-name",
        help="Override the per-run subdirectory name (default: cfg.name).",
    )
    parser.add_argument(
        "--timestamped",
        action="store_true",
        help="Append a UTC timestamp suffix to the run name.",
    )
    parsed = parser.parse_args(argv)

    cfg_path = parsed.config
    if cfg_path is None and parsed.args and _looks_like_path(parsed.args[0]):
        cfg_path = parsed.args[0]

    if parsed.pipeline:
        from blockference.pipeline import run_pipeline

        if not cfg_path:
            parser.error("--pipeline requires a config path")
        cfg = load_experiment_config(cfg_path)
        if parsed.engine:
            cfg.engine = parsed.engine
        result = run_pipeline(
            cfg,
            output_root=parsed.output_root,
            run_name=parsed.run_name,
            timestamped=parsed.timestamped,
        )
        print(f"[pipeline] artefacts at: {result.paths.run_dir}")
        print(f"[pipeline] schema check ok: {result.schema_report.ok}")
        print(f"[pipeline] artefacts check ok: {result.artefacts_report.ok}")
        return 0 if (result.schema_report.ok and result.artefacts_report.ok) else 1

    if cfg_path:
        cfg = load_experiment_config(cfg_path)
        if parsed.output:
            cfg.output.path = parsed.output
        if parsed.engine:
            cfg.engine = parsed.engine
        run_experiment(cfg)
        return 0

    if len(parsed.args) >= 3:
        run_grid(
            dimension=parsed.args[0],
            no_agents=parsed.args[1],
            no_timesteps=parsed.args[2],
            output_path=parsed.output or "result.csv",
            engine=parsed.engine or "radcad",
        )
        return 0

    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
