# Architecture

This is a tour of the `blockference` package, top-down. For *what* AIF
is, see [`theory.md`](theory.md). For *how to use* the public surface,
see [`api.md`](api.md).

## Layered view

```
┌─────────────────────────────────────────────────────────────────────────┐
│  CLI / notebooks / scripts                                              │
│    python -m blockference.simulations.grid_sim --pipeline configs/...   │
└─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  blockference.pipeline.run_pipeline  ←─ composes everything below       │
└─────────────────────────────────────────────────────────────────────────┘
            │                   │                  │                │
            ▼                   ▼                  ▼                ▼
┌─────────────────────┐ ┌──────────────────┐ ┌──────────────┐ ┌──────────┐
│ blockference.config │ │ blockference.    │ │ blockference │ │ blockf.  │
│  ExperimentConfig   │ │    simulations   │ │     .viz     │ │   .io    │
│  load_experiment_…  │ │  run_experiment  │ │  plots,      │ │  layout, │
└─────────────────────┘ │  run_grid        │ │  animation   │ │  persist,│
                        └──────────────────┘ └──────────────┘ │  validate│
                                  │                            └──────────┘
                                  ▼
                ┌─────────────────────────────────────┐
                │ blockference.envs        (dynamics) │
                │ blockference.gridference (gen-model)│
                │ blockference.utils       (math+cad) │
                └─────────────────────────────────────┘
                                  │
                                  ▼
                ┌─────────────────────────────────────┐
                │ pymdp + numpy + cadCAD/radCAD       │
                └─────────────────────────────────────┘
```

Higher layers may import from lower layers. The reverse is forbidden.

## Module responsibilities

### `blockference.gridference`

Owns the generative-model class `ActiveGridference` and the per-step
inference + planning functions:

* `actinf_planning_single(agent, env_state, A, B, C, prior)` — one step.
* `actinf_graph(agent_network)` — one step per node of a `networkx`
  graph.
* `_move(action_id, env_state, border)` — the canonical 5-action grid
  transition; reused from `policy.py`.
* `make_grid(grid_len, grid_dim=2)` — coordinate-list helper.

### `blockference.agent`

`BlockferenceAgent` (and the back-compat alias `Agent`) is a thin
subclass of `pymdp.agent.Agent`. It exists so that downstream code can
attach extra behaviour without monkey-patching upstream.

### `blockference.envs`

Environment classes encapsulate **world dynamics** (state transitions,
collisions, observation generation):

* `GridAgent` — single-/multi-agent grid with simple step semantics.
* `TwoMultiGridAgent` — two-agent grid with collision handling and
  relative-position observations.

### `blockference.simulations`

Runnable experiments. Each module:

1. Builds an initial state (`agents`, `priors`, ...).
2. Defines cadCAD `policies` and `variables` blocks.
3. Calls `Simulation(...).run()`.
4. Returns the result as a `pandas.DataFrame`.

`grid_sim.run_grid(dimension, no_agents, no_timesteps, output_path)` is
the worked example.

### `blockference.utils`

* `utils.py` — math (`infer_states`, `entropy`, `kl_divergence`,
  `calculate_G`, `calculate_G_policies`, `compute_prob_actions`) and
  plotting helpers.
* `policy.py` — cadCAD policy functions (`p_actinf_single`,
  `p_actinf_dict`).
* `mutual_info.py` — CLI for analysing simulation traces with
  `sklearn.feature_selection.mutual_info_regression`.

## State variables in cadCAD experiments

The standard five state variables a multi-agent grid experiment carries:

| Variable     | Type                              | What it holds                                  |
|--------------|-----------------------------------|------------------------------------------------|
| `agents`     | `dict[int, ActiveGridference]`    | The agent objects themselves.                  |
| `priors`     | `dict[int, np.ndarray]`           | Per-agent prior over hidden states.            |
| `env_states` | `dict[int, tuple[int, int]]`      | Per-agent `(y, x)` location.                   |
| `actions`    | `dict[int, int]`                  | Per-agent last action index (or `''` initial). |
| `inferences` | `dict[int, np.ndarray]`           | Per-agent posterior over hidden states.        |

cadCAD writes one row per timestep × substep × run. See
`blockference/simulations/grid_sim.py` for the precise update functions.

## Coordinate convention

`(y, x)` everywhere. This matches `numpy` row-major and avoids the
`(x, y)` ambiguity that bit several earlier notebooks.

## Data flow inside one cadCAD timestep

```
previous_state              policy (p_actinf_dict)
  ─ agents                    ┐
  ─ priors                    │ for each agent:
  ─ env_states  ──────────────┤   infer_states(...)
  ─ actions                   │   calculate_G_policies(...)
  ─ inferences                │   softmax / sample
                              ┘   _move → new env state
                                  emit update
                                  │
                                  ▼
                       state-update functions
                       (s_agents, s_priors, ...)
                                  │
                                  ▼
                          next_state
```
