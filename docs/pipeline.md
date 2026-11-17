# Pipeline contract

## Configuration

YAML and TOML use the same strict schema:

```yaml
name: smoke
seed: 0
engine: radcad
grid:
  dimension: 3
  planning_length: 1
  max_policies: 100000
  affordances: [UP, DOWN, LEFT, RIGHT, STAY]
simulation:
  timesteps: 2
  runs: 1
  n_agents: 1
  target: [2, 2]
  initial_state: [0, 0]
  # Optional for multi-agent runs:
  # initial_states: [[0, 0], [2, 2]]
output:
  path: results/smoke.csv
```

Unknown keys, unsafe names, invalid coordinates, unsupported actions, invalid
engines, malformed paths, and an incorrectly sized or duplicate `initial_states`
list are rejected. A multi-agent run (`n_agents > 1`) requires one distinct
`simulation.initial_states` coordinate per agent; the legacy `initial_state`
broadcast default is valid only for a single agent and otherwise fails fast
with an explicit error before any simulation backend runs.

File roots must be mappings. Empty/comment-only YAML and explicit YAML `null`
retain the default-configuration contract; `false`, zero, an empty string, and
lists are not treated as omitted configuration. They fail before output setup.

## Required run tree

```text
RUN/
├── config.yml
├── run.log
├── data/
│   ├── trajectory.csv
│   ├── summary.json
│   ├── generative_model.json
│   ├── generative_model.npz
│   ├── policies.json
│   └── per_step.csv
├── manifest.json
├── viz/
│   ├── trajectory.png
│   ├── action_distribution.png
│   ├── efe.png
│   ├── belief_heatmap_run_<run>_agent_<agent>_first.png
│   └── belief_heatmap_run_<run>_agent_<agent>_last.png
├── animations/trajectory.gif
└── validation_report.json
```

Parquet files may be added when an installed parquet engine supports the
serialised schema. They are never used in place of required CSV/JSON files.

## Run and agent identity

Model and policy JSON use version 2:
`{"version": 2, "runs": {"<run>": {"<agent>": "<record>"}}}`.
Every configured `(run, agent)` owns its model and ordered policy set. NPZ
keys are `run_<run>/agent_<agent>/<matrix>`. Trajectory agent records bind
`agent_id` and `model_sha256` to that same immutable model; a refreshed
manifest cannot certify swapped JSON/NPZ models.

Summary version 2 retains aggregate row/agent/grid/run counts and places
`final_positions`, `trajectory_lengths`, and `belief_history_lengths` under
`runs[<run>]`. Old flat summary fields and version-1 model/policy artifacts
are not silently migrated. Preserve historical trees and generate a new run.

Each run contains `(timestep=0, substep=0)` followed by
`(timestep=1..T, substep=1)`. Every run retains its configured starts, complete
unique per-agent diagnostics, ordered affordances, and actual planning cap.
Actions and observation indices are integer categories, not numeric strings,
floats, or booleans. Policy diagnostics have `actions**planning_length`
entries; action distributions have action cardinality.

Plots, heatmaps, and animation histories are keyed by `(run, agent)` and
ordered by actual time/substep. Independent runs are not concatenated.
Both heatmaps shown above are required for every configured pair.

## Validation

The aggregate `ValidationReport.ok` is the conjunction of trajectory schema,
generative-model invariants, complete per-step diagnostics, parsed artefacts,
and rendered files. `PipelineResult.ok` returns that value, and the `pipeline`
CLI exits with status 1 when it is false.
Missing parsed prerequisites, malformed policies or matrices, invalid UTF-8,
and mismatched keyed diagnostics produce failed reports, not successful
skipped boundaries. The validation CLI emits its JSON report and status 1.

These verdicts certify software and artefact integrity only. They do not — and
cannot — certify the empirical adequacy of a scientific hypothesis. Any
adequacy claim must carry its own independent datasets, provenance, and
sensitivity analyses documented separately from `PipelineResult.ok`.

`manifest.json` records SHA-256 digests and byte sizes for every required
stable artifact. It is written after persistence and rendering complete; the
validator checks it independently. `validation_report.json` and `run.log` are
mutable metadata and are deliberately excluded from the manifest digest.

The complete local release gate is
`uv run python scripts/release_check.py --arm all`; the no-argument default is
the fast selection. See [`gate.md`](gate.md).
