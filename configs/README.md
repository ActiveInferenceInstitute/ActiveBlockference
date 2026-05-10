# `configs/` — experiment configurations

YAML/TOML files describing complete cadCAD experiments. The schema is
declared by :class:`blockference.config.ExperimentConfig` and validated
when the file is loaded.

| File              | Purpose                                              |
|-------------------|------------------------------------------------------|
| `example.yml`     | Reference YAML config; documents every supported key.|
| `smoke.yml`       | Tiniest valid experiment — used by CI smoke test.    |
| `sweep.toml`      | Same shape in TOML; proves both loaders work.        |

## Running

```bash
uv run python -m blockference.simulations.grid_sim configs/example.yml
uv run python -m blockference.simulations.grid_sim --config configs/sweep.toml --output out/sweep.csv
```

Programmatic equivalent:

```python
from blockference.config import load_experiment_config
from blockference.simulations import run_experiment

cfg = load_experiment_config("configs/example.yml")
df = run_experiment(cfg)
```

## Schema

```yaml
name: <string>                 # human label written into logs
seed: <int | null>             # RNG seed (random.seed); null → non-deterministic
grid:
  dimension: <int >= 2>        # side-length of square grid
  planning_length: <int >= 1>  # cadCAD policy horizon
  affordances: [<string>, ...] # action labels (default 5-action set)
simulation:
  timesteps: <int >= 1>
  runs: <int >= 1>             # cadCAD Monte-Carlo runs
  n_agents: <int >= 1>
  target: random | [y, x]      # preferred goal cell, per agent
  initial_state: [y, x]
output:
  path: <string | null>        # CSV output path; null disables write
```

See [`AGENTS.md`](AGENTS.md) for editing rules.
