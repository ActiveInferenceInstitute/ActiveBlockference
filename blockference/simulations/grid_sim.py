"""radCAD/cadCAD-driven multi-agent grid simulation."""

from __future__ import annotations

import contextlib
import io
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from blockference.actions import DEFAULT_AFFORDANCES
from blockference.config import ExperimentConfig
from blockference.gridference import ActiveGridference, make_grid
from blockference.utils.policy import p_actinf_dict

__all__ = [
    "PER_STEP_FIELDS",
    "build_initial_state",
    "build_state_update_blocks",
    "run_experiment",
    "run_grid",
]


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


def build_initial_state(
    grid,
    n_agents: int,
    dimension: int,
    target,
    initial_state,
    planning_length: int,
    affordances=DEFAULT_AFFORDANCES,
    max_policies: int = 100_000,
    initial_states: Sequence[Sequence[int]] | None = None,
    rng: np.random.Generator | None = None,
):
    """Construct the initial cadCAD state for a multi-agent grid world.

    ``initial_states`` makes each agent's starting coordinate explicit. When it
    is omitted, the legacy ``initial_state`` coordinate is used for every
    agent. Random targets are drawn from the supplied experiment-local
    generator, so this function does not mutate process-global RNG state.
    """

    if len(grid) != int(dimension) ** 2:
        raise ValueError("grid must contain exactly dimension squared coordinates")
    if isinstance(n_agents, bool) or not isinstance(n_agents, int) or n_agents < 1:
        raise ValueError("n_agents must be a positive integer")
    generator = np.random.default_rng() if rng is None else rng
    if initial_states is None:
        starts = [tuple(initial_state)] * n_agents
    else:
        starts = [tuple(value) for value in initial_states]
        if len(starts) != n_agents:
            raise ValueError("initial_states must contain exactly one coordinate per agent")
        if len(set(starts)) != len(starts):
            raise ValueError("initial_states must contain unique coordinates")
    state: dict[str, Any] = {"agents": {}, **{key: {} for key, _ in PER_STEP_FIELDS}}
    for agent_id in range(int(n_agents)):
        agent = ActiveGridference(
            grid,
            planning_length=planning_length,
            affordances=affordances,
            max_policies=max_policies,
        )
        if target == "random":
            random_target = generator.integers(0, int(dimension), size=2)
            preferred = (int(random_target[0]), int(random_target[1]))
        else:
            preferred = (int(target[0]), int(target[1]))
        agent.get_C(preferred)
        agent.get_D(starts[agent_id])  # type: ignore[arg-type]
        state["agents"][agent_id] = agent
        assert agent.D is not None
        state["priors"][agent_id] = agent.D.copy()
        state["env_states"][agent_id] = agent.env_state
        state["inferences"][agent_id] = agent.current_inference
        state["actions"][agent_id] = agent.current_action
        for key, _ in PER_STEP_FIELDS[4:]:
            state[key][agent_id] = None
    return state


def _make_state_updater(state_key: str, update_key: str):
    """Return a cadCAD state updater for one diagnostic field."""

    def state_update(params, substep, state_history, previous_state, policy_input):
        new = previous_state[state_key].copy()
        for update in policy_input.get("agent_updates", []):
            new[update["source"]] = update[update_key]
        return state_key, new

    state_update.__name__ = f"s_{state_key}"
    return state_update


def _agents_state_update(params, substep, state_history, previous_state, policy_input):
    agents_new = previous_state["agents"].copy()
    for update in policy_input.get("agent_updates", []):
        agent = agents_new[update["source"]]
        agent.prior = update["update_prior"]
        agent.env_state = update["update_env"]
        agent.current_action = update["update_action"]
        agent.current_inference = update["update_inference"]
    return "agents", agents_new


def build_state_update_blocks(
    grid, *, rng: np.random.Generator | None = None
) -> list[dict[str, Any]]:
    """Assemble the canonical multi-agent state-update block."""

    def policy(params, substep, state_history, previous_state):
        return p_actinf_dict(params, substep, state_history, previous_state, grid, rng=rng)

    variables = {"agents": _agents_state_update}
    for key, update_key in PER_STEP_FIELDS:
        variables[key] = _make_state_updater(key, update_key)
    return [{"policies": {"p_actinf": policy}, "variables": variables}]


def _run_radcad(
    initial_state: dict[str, Any],
    state_update_blocks: list[dict[str, Any]],
    params: dict[str, Any],
    timesteps: int,
    runs: int,
) -> pd.DataFrame:
    from radcad import Model, Simulation

    model = Model(
        initial_state=initial_state,
        state_update_blocks=state_update_blocks,
        params=params,
    )
    return Simulation(model=model, timesteps=int(timesteps), runs=int(runs)).run()


def _run_cadcad(
    initial_state: dict[str, Any],
    state_update_blocks: list[dict[str, Any]],
    params: dict[str, Any],
    timesteps: int,
    runs: int,
) -> pd.DataFrame:
    """Drive the same state-update blocks through cadCAD."""

    from cadCAD.configuration import Experiment
    from cadCAD.configuration.utils import config_sim
    from cadCAD.engine import ExecutionContext, ExecutionMode, Executor

    sim_config = config_sim({"N": int(runs), "T": range(int(timesteps)), "M": params})
    experiment = Experiment()
    experiment.append_configs(
        initial_state=initial_state,
        partial_state_update_blocks=state_update_blocks,
        sim_configs=sim_config,
    )
    executor = Executor(
        exec_context=ExecutionContext(context=ExecutionMode().local_mode),
        configs=experiment.configs,
    )
    # cadCAD prints a banner and progress bars directly to stdout. Library
    # callers and the CLI promise structured output, so keep that backend
    # chatter inside the adapter boundary.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        raw_result, _tensor_field, _sessions = executor.execute()
    return raw_result


def run_experiment(cfg: ExperimentConfig) -> pd.DataFrame:
    """Run a validated configuration and optionally write its trajectory CSV.

    Each configured run receives an independent child stream derived from the
    experiment seed. This makes run ``1`` invariant when later runs are added
    and keeps both simulation backends on the same random schedule.
    """

    if not isinstance(cfg, ExperimentConfig):
        raise TypeError("cfg must be an ExperimentConfig")
    grid = make_grid(cfg.grid.dimension, 2)
    run_seeds = np.random.SeedSequence(cfg.seed).spawn(cfg.simulation.runs)
    frames: list[pd.DataFrame] = []
    for run_index, run_seed in enumerate(run_seeds, start=1):
        rng = np.random.default_rng(run_seed)
        initial_state = build_initial_state(
            grid=grid,
            n_agents=cfg.simulation.n_agents,
            dimension=cfg.grid.dimension,
            target=cfg.simulation.target,
            initial_state=cfg.simulation.initial_state,
            planning_length=cfg.grid.planning_length,
            affordances=cfg.grid.affordances,
            max_policies=cfg.grid.max_policies,
            initial_states=cfg.simulation.initial_states,
            rng=rng,
        )
        params = {"preferred_state": [grid], "initial_state": [grid], "noise": [0]}
        state_update_blocks = build_state_update_blocks(grid, rng=rng)
        if cfg.engine == "radcad":
            result = _run_radcad(
                initial_state,
                state_update_blocks,
                params,
                cfg.simulation.timesteps,
                1,
            )
        elif cfg.engine == "cadcad":
            result = _run_cadcad(
                initial_state,
                state_update_blocks,
                params,
                cfg.simulation.timesteps,
                1,
            )
        else:
            raise ValueError(f"unknown engine {cfg.engine!r}")
        frame = pd.DataFrame(result)
        frame["run"] = run_index
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    if cfg.output.path:
        output = Path(cfg.output.path)
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, index=False)
    return frame


def run_grid(
    dimension: int,
    n_agents: int,
    timesteps: int,
    *,
    output_path: str | None = "result.csv",
    planning_length: int = 2,
    max_policies: int = 100_000,
    target: str | tuple[int, int] = "random",
    initial_state: tuple[int, int] = (0, 0),
    initial_states: Sequence[Sequence[int]] | None = None,
    affordances=DEFAULT_AFFORDANCES,
    seed: int | None = None,
    runs: int = 1,
    engine: str = "radcad",
) -> pd.DataFrame:
    """Build and run an :class:`ExperimentConfig` from typed primitives."""

    cfg = ExperimentConfig.from_dict(
        {
            "name": "run_grid",
            "seed": seed,
            "engine": engine,
            "grid": {
                "dimension": dimension,
                "planning_length": planning_length,
                "max_policies": max_policies,
                "affordances": list(affordances),
            },
            "simulation": {
                "timesteps": timesteps,
                "runs": runs,
                "n_agents": n_agents,
                "target": target,
                "initial_state": list(initial_state),
                **(
                    {"initial_states": [list(value) for value in initial_states]}
                    if initial_states is not None
                    else {}
                ),
            },
            "output": {"path": output_path},
        }
    )
    return run_experiment(cfg)
