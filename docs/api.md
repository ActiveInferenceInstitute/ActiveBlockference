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
  `simulation.initial_state` remains the single-agent compatibility default;
  for `n_agents > 1` provide `simulation.initial_states` with one distinct
  start coordinate per agent (a multi-agent run without it fails fast).
* `blockference.envs.DiscreteEnvironment` is the runtime-checkable environment
  protocol; `GridWorld` conforms and adds lossless `serialize()` / `load()`.
* `blockference.io.atomic_replace(path, writer)` publishes any file atomically;
  `parse_trajectory_records(df)` normalizes a trajectory into typed, ordered
  `TrajectoryRecord` values for backend/persistence parity.

Agent IDs are built-in `int` or `str`. Version-2 environment serialization
preserves typed IDs, insertion order, and positions: integer `1` and string
`"1"` remain distinct. Sequence actions follow registry order; conflict
priority remains `(type name, repr)`, not numeric or insertion order.
Unsupported Hashables, duplicate IDs, malformed typed records, and lossy
legacy string-key payloads are rejected.

`norm_dist` and `sample` scale finite nonnegative weights before summation.
Normalization retains uniform zero slices; sampling rejects zero total
before consuming RNG. A supplied `Generator` owns sampling; omitted `rng`
retains the global NumPy stream. Neither helper mutates its input.

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

Run the complete local release gate with
`uv run python scripts/release_check.py --arm all`; the no-argument default is
the fast selection. See [`gate.md`](gate.md).

## Persistence and validation

Persistence is run/agent keyed. Pass
`persist_dataframe(..., models={(run, agent_id_string): agent})`; use the
same keys for `persist_generative_model`, `persist_generative_model_npz`,
and `persist_policies`. Model records retain ordered action labels,
dimensions, border, policy length, and the actual policy cap.

Focused model, policy, and per-step validators require their parsed
configuration/model/trajectory prerequisites. Missing prerequisites fail.
The version-2 artifact and summary layouts, schedules, and required renders
are defined in [`pipeline.md`](pipeline.md); refreshing hashes never replaces
identity or value validation.

## GNN, manuscript, and notebook consumers

`blockference.gnn.spec_arrays` exports a numeric uniform prior over complete
policies; runtime `ActiveGridference.E` remains the ordered action vocabulary.
For five actions and horizon two, exported `E`, `pi`, and `G` have 25 entries,
while `B` has five action slices. Float text preserves round-trip precision,
including one-element vectors. The pinned parser canonicalizes `pi` to `π`;
parsed-payload consumers use that canonical name.

The external worker and independently reviewed resources are described in
[`gnn.md`](gnn.md). Pin inspection and `gnn_*` manuscript metadata do not
constitute source execution, installed-origin, or scientific qualification.

`load_manuscript_experiment` reads the confined repository-relative
`experiment.config_path` in manuscript metadata through the typed YAML/TOML
loader. Missing or escaping selections do not fall back to a default file.
See [`manuscript/README.md`](manuscript/README.md) for relocated resources.

`blockference.notebook_inventory` owns `tracked_notebooks`,
`validate_notebook`, `validate_all`, and `execute_notebooks`.
`validate_all` returns `(frozen_paths, failures)`; execution validates and
executes that frozen tracked selection in its supplied workspace.
See [`../notebooks/README.md`](../notebooks/README.md) for selection and kernel
boundaries.

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
Histories are run/agent keyed. Select both `run` and `agent_id` for a heatmap
when an agent occurs in multiple runs. Its `timestep` argument indexes the
ordered history; titles and GIF labels report actual run/time/substep values.
