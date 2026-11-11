# ActiveBlockference

ActiveBlockference is a CPU-oriented Active Inference library for deterministic,
discrete-state radCAD and cadCAD grid simulations. It provides strict YAML/TOML
configuration, canonical `(y, x)` dynamics, persisted diagnostics, rendered
artefacts, and fail-closed validation.

[![CI](https://github.com/ActiveInferenceInstitute/ActiveBlockference/actions/workflows/ci.yml/badge.svg)](https://github.com/ActiveInferenceInstitute/ActiveBlockference/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

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

For single-agent runs, `simulation.initial_state` remains the compatibility
default. For multi-agent runs (`n_agents > 1`) the simultaneous-transition
environment requires a unique starting cell per agent, so use
`simulation.initial_states` with one distinct coordinate per agent; a
multi-agent run without it fails fast with an explicit error. A seeded run owns
its random generator and does not alter the caller's global random state.

The grid implementation is self-contained. Install `active-blockference[pymdp]`
only when using the optional `BlockferenceAgent` adapter; install
`active-blockference[research]` for the explicitly selected OpenAI GRTs
provider. Offline tests and the release gate never require either extra.

## Reliability and contracts

* All JSON, CSV, NPZ, config, model, and render artefacts are published
  **atomically** through a sibling temporary file and renamed into place, so an
  interrupted run can never leave a partially written file (a leftover temp file
  is flagged as drift by validation).
* `ValidationReport.ok` and `PipelineResult.ok` certify **software and artefact
  integrity only** — never the empirical adequacy of a scientific hypothesis.
* `blockference.io.parse_trajectory_records` normalizes a persisted trajectory
  into typed, deterministically ordered `TrajectoryRecord` values so backend
  (radCAD vs cadCAD) and persistence round-trips are compared on values, not
  stringified cells.
* `GridWorld` conforms to the `blockference.envs.DiscreteEnvironment` protocol
  (state identity, observations, simultaneous actions, collision resolution,
  and lossless `serialize`/`load`), so future environments can be added behind
  the same contract instead of special-casing the grid.

## Python API

```python
from blockference import ExperimentConfig, GridWorld, run_pipeline

config = ExperimentConfig.from_dict({
    "name": "demo",
    "seed": 42,
    "grid": {"dimension": 3, "planning_length": 2,
             "affordances": ["UP", "RIGHT", "STAY"]},
    "simulation": {"timesteps": 10, "runs": 1, "n_agents": 2,
                    "target": [2, 2],
                    "initial_states": [[0, 0], [1, 1]]},
})
result = run_pipeline(config)
assert result.ok

world = GridWorld(3, {0: (0, 0), 1: (1, 1)})
world.step({0: 1, 1: 0})
```

## Repository map

* `blockference/` — installable library and CLI.
* `configs/` — strict YAML/TOML examples.
* `specs/` — the GNN declaration of the grid-world model and its pinned parser
  record.
* `docs/` — architecture, theory, API, and pipeline contracts.
* `notebooks/` — executable current-API teaching notebooks.
* `GRTs/` — optional provider-based local research workflow.
* `tests/` — unit, integration, artefact, and notebook checks.

## Generative-model specification

The grid-world generative model is declared in
[`specs/gridworld_3x3.gnn.md`](specs/gridworld_3x3.gnn.md) using
[Generalized Notation Notation](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation)
(GNN). The document is generated from the implementation and validated against
the upstream parser pinned in [`specs/gnn-pin.json`](specs/gnn-pin.json)
(GNN v5.0.3, commit `8f293d9ba`): the validator reads the declared parameters
back, builds the runtime model from them, and requires the declared matrices to
match exactly. See [`docs/gnn.md`](docs/gnn.md) for the commands and the
conformance contract.

GNN maintains a [downstream adoption
guide](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/blob/main/docs/gnn/integration/downstream_adoption.md)
for repositories that consume the notation.

See [`docs/README.md`](docs/README.md) for the documentation index,
[`docs/api.md`](docs/api.md), [`docs/pipeline.md`](docs/pipeline.md), and
[`docs/development.md`](docs/development.md) for the complete contracts and
workflow. Contributions are described in [`docs/contributing.md`](docs/contributing.md);
security issues are reported privately per [`SECURITY.md`](SECURITY.md); the
repository is released under the [MIT License](LICENSE).

## InstituteOS sidecar

This repository carries a committed [`.aii/`](.aii/README.md) sidecar with curated
InstituteOS metadata (`aii-sidecar/v1`): category, maturity, ecosystem relations,
key artifacts, provenance, and portable tasks. It is read by
[InstituteOS](https://github.com/ActiveInferenceInstitute/InstituteOS) tooling and
does not affect the project's build or runtime.
