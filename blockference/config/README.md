# `blockference/config/` — typed experiment configuration

Loads YAML/TOML files into validated dataclasses.

## Surface

| Symbol                     | Kind     | Purpose                                     |
|----------------------------|----------|---------------------------------------------|
| `ExperimentConfig`         | dataclass| Top-level config (`name`, `seed`, ...).     |
| `GridConfig`               | dataclass| Generative-model parameters.                |
| `SimulationConfig`         | dataclass| cadCAD timesteps / runs / agents.           |
| `OutputConfig`             | dataclass| Output paths.                               |
| `load_experiment_config()` | function | Read a `.yml` / `.yaml` / `.toml` file.     |

## Example

```python
from blockference.config import load_experiment_config
cfg = load_experiment_config("configs/example.yml")
print(cfg.simulation.n_agents, cfg.output.path)
```

## Validation

Each dataclass enforces invariants in `__post_init__`:

* `dimension >= 2`, `planning_length >= 1`, and `max_policies >= 1`.
* `timesteps >= 1`, `runs >= 1`, `n_agents >= 1`.
* List fields read from YAML are coerced to tuples where the schema
  expects coordinates.
* `simulation.initial_states` must contain one distinct coordinate per agent.
  The `initial_state` compatibility default applies only to a single agent.

If validation fails, the loader raises `ValueError` *immediately*; the
caller never sees a partially-valid config.

YAML/TOML roots must be mappings. Empty/comment-only YAML and explicit `null`
retain default configuration; `false`, zero, empty-string and list roots do
not. Nested null compatibility does not weaken unknown-key rejection.

`load_manuscript_experiment` resolves the metadata's nonempty
`experiment.config_path` inside the selected repository and uses this same
typed loader. Missing, absolute, or escaping selections fail without a
hardcoded manuscript-config fallback.

See [`AGENTS.md`](AGENTS.md) for editing rules and the
[`configs/README.md`](../../configs/README.md) for the user-facing schema.
