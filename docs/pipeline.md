# The pipeline

`blockference.pipeline.run_pipeline` composes the full
**config → simulate → persist → visualise → animate → validate** flow.
This document explains each step, where it lives, and how to run only
parts of the chain.

## End-to-end

```bash
uv run python -m blockference.simulations.grid_sim --pipeline configs/example.yml
```

This produces:

```
output/<run_name>/
├── config.yml                       ← copy of the source config
├── run.log                          ← structured per-run log
├── data/
│   ├── trajectory.csv               ← cadCAD result frame
│   ├── trajectory.parquet           ← (when pyarrow is installed)
│   ├── summary.json                 ← n_agents, final positions, …
│   ├── generative_model.json        ← A/B/C/D/E for every agent
│   ├── per_step.csv                 ← per-(agent, t) action / posterior / EFE proxy
│   └── policies.json                ← enumerated policy set
├── viz/
│   ├── trajectory.png
│   ├── action_distribution.png
│   ├── efe_proxy.png
│   ├── belief_heatmap_first.png
│   └── belief_heatmap_last.png
├── animations/
│   └── trajectory.gif
└── validation_report.json           ← schema + math + artefact checks
```

## The seven steps

| # | Step                            | Module                                                  |
|---|---------------------------------|---------------------------------------------------------|
| 1 | Load + validate config          | `blockference.config.load_experiment_config`            |
| 2 | Build typed output paths        | `blockference.io.build_run_paths`                       |
| 3 | Persist config (round-trip)     | `blockference.io.persist_config`                        |
| 4 | Run cadCAD/radCAD experiment    | `blockference.simulations.run_experiment`               |
| 5 | Persist data + summary          | `blockference.io.persist_dataframe`, `persist_summary`  |
| 6 | Render plots + animation        | `blockference.viz.{plots,animation}`                    |
| 7 | Validate artefacts              | `blockference.io.validate_run_outputs`                  |
| 8 | Validate generative-model math  | `blockference.io.validate_generative_model`             |
| 9 | Persist matrices, policies, per-step | `persist_generative_model`, `persist_policies`, `persist_per_step_records` |
| 10| Structured per-run logging      | `blockference.logging_setup.configure_run_logging`      |

## Composability — run only some steps

```python
from blockference.config import load_experiment_config
from blockference.io import build_run_paths, validate_run_outputs
from blockference.simulations import run_experiment
from blockference.viz import plot_trajectory, animate_trajectory
import pandas as pd

# 1. just the simulation
cfg = load_experiment_config("configs/example.yml")
df  = run_experiment(cfg)

# 2. just visualisation against an existing CSV
df  = pd.read_csv("output/gridworld_example/data/trajectory.csv")
plot_trajectory(df, "out/trajectory.png")
animate_trajectory(df, "out/trajectory.gif")

# 3. just validation against an existing run dir
report = validate_run_outputs(build_run_paths("output", "gridworld_example"))
print(report.to_dict())
```

## Schema validation

Two layers:

1. **Config schema** — enforced at load time by
   `blockference.config.experiment` dataclass `__post_init__` methods.
2. **Trajectory + artefact schema** — enforced after the run by
   `blockference.io.validate_trajectory_dataframe` and
   `blockference.io.validate_run_outputs`. Failures are written into
   `validation_report.json` so downstream consumers can gate on it.

## CI

The smoke configuration `configs/smoke.yml` runs the full pipeline in
under 2 seconds and is exercised on every PR via
`.github/workflows/ci.yml`.
