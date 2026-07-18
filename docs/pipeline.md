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
engines, malformed paths, and an incorrectly sized `initial_states` list are
rejected before execution. If `initial_states` is omitted, the legacy
`initial_state` coordinate is used for every agent.

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
│   └── efe.png
├── animations/trajectory.gif
└── validation_report.json
```

Parquet files may be added when an installed parquet engine supports the
serialised schema. They are never used in place of required CSV/JSON files.

## Validation

The aggregate `ValidationReport.ok` is the conjunction of trajectory schema,
generative-model invariants, complete per-step diagnostics, parsed artefacts,
and rendered files. `PipelineResult.ok` returns that value, and the `pipeline`
CLI exits with status 1 when it is false.

`manifest.json` records SHA-256 digests and byte sizes for every required
stable artifact. It is written after persistence and rendering complete; the
validator checks it independently. `validation_report.json` and `run.log` are
mutable metadata and are deliberately excluded from the manifest digest.

The complete local release gate is `uv run python scripts/release_check.py`.
