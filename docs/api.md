# API

## Core objects

* `blockference.GridWorld(dimension, positions, affordances=...)` stores
  `(y, x)` positions and applies deterministic simultaneous actions.
* `blockference.ActiveGridference(grid, planning_length, env_state,
  affordances, max_policies)` owns `A`, `B`, `C`, `D`, and `E`. Updating `E`
  rebuilds `B`; `max_policies` prevents unbounded policy enumeration.
* `blockference.BlockferenceAgent(A, B, **kwargs)` is the optional upstream
  pymdp adapter. Install `active-blockference[pymdp]` to use it.
* `blockference.ExperimentConfig.from_dict(mapping)` rejects unknown keys and
  validates dimensions, coordinates, action labels, engine, and paths.
  `simulation.initial_states` can provide one start coordinate per agent;
  otherwise `simulation.initial_state` is broadcast.

## Execution

```python
from blockference import ExperimentConfig, run_pipeline

config = ExperimentConfig.from_dict({
    "name": "demo",
    "seed": 7,
    "grid": {"dimension": 3, "planning_length": 1},
    "simulation": {"timesteps": 4, "runs": 1, "n_agents": 1,
                    "target": "random", "initial_state": [0, 0]},
})
result = run_pipeline(config, output_root="output")
print(result.ok, result.paths.run_dir)
```

`run_grid(dimension, n_agents, timesteps, *, output_path, planning_length,
max_policies, target, initial_state, initial_states, affordances, seed, runs,
engine)` is the typed primitive wrapper. Engine names are `radcad` and `cadcad`.

Run the complete local release gate with `uv run python scripts/release_check.py`.

## CLI

```text
blockference run --config CONFIG
blockference pipeline --config CONFIG [--output-root ROOT] [--run-name NAME]
blockference validation --run-dir RUN_DIRECTORY
```

## Visualisation

`plot_trajectory`, `plot_action_distribution`, `plot_belief_heatmap`,
`plot_efe`, and `animate_trajectory` return the written path. `plot_efe`
consumes the persisted `efe` vectors and plots the minimum policy EFE per step.
