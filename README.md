# ActiveBlockference

ActiveBlockference is a CPU-oriented Active Inference library for deterministic,
discrete-state radCAD and cadCAD grid simulations. It provides strict YAML/TOML
configuration, canonical `(y, x)` dynamics, persisted diagnostics, rendered
artefacts, and fail-closed validation.

## Quick start

```bash
uv sync --locked --extra dev
uv run pytest
uv run ruff check blockference tests
uv run python scripts/release_check.py
uv run blockference pipeline --config configs/smoke.yml --output-root output
```

The CLI has three explicit commands:

```bash
blockference run --config configs/example.yml
blockference pipeline --config configs/example.yml --output-root output
blockference validation --run-dir output/gridworld_example
```

The pipeline emits `config.yml`, CSV trajectory and per-step diagnostics, JSON
summary/model/policies, an NPZ model archive, PNG visualisations, a GIF, a log,
`manifest.json`, and one aggregate `validation_report.json`. Parquet is an
optional additional format; CSV and JSON remain required. A non-empty run
directory is refused unless `--reuse` is supplied explicitly.

For multi-agent runs, `simulation.initial_state` remains the compatibility
default and is broadcast to every agent. Use `simulation.initial_states` to
record distinct starts explicitly. A seeded run owns its random generator and
does not alter the caller's global random state.

The grid implementation is self-contained. Install `active-blockference[pymdp]`
only when using the optional `BlockferenceAgent` adapter; install
`active-blockference[research]` for the explicitly selected OpenAI GRTs
provider. Offline tests and the release gate never require either extra.

## Python API

```python
from blockference import ExperimentConfig, GridWorld, run_pipeline

config = ExperimentConfig.from_dict({
    "name": "demo",
    "seed": 42,
    "grid": {"dimension": 3, "planning_length": 2,
             "affordances": ["UP", "RIGHT", "STAY"]},
    "simulation": {"timesteps": 10, "runs": 1, "n_agents": 2,
                    "target": [2, 2], "initial_state": [0, 0]},
})
result = run_pipeline(config)
assert result.ok

world = GridWorld(3, {0: (0, 0), 1: (1, 1)})
world.step({0: 1, 1: 0})
```

## Repository map

* `blockference/` — installable library and CLI.
* `configs/` — strict YAML/TOML examples.
* `docs/` — architecture, theory, API, and pipeline contracts.
* `notebooks/` — executable current-API teaching notebooks.
* `GRTs/` — optional provider-based local research workflow.
* `tests/` — unit, integration, artefact, and notebook checks.

See [`docs/api.md`](docs/api.md), [`docs/pipeline.md`](docs/pipeline.md), and
[`docs/development.md`](docs/development.md) for the complete contracts.
