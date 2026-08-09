# ActiveBlockference

ActiveBlockference is a CPU-oriented Python library for discrete Active
Inference experiments in radCAD and cadCAD. It provides deterministic grid
worlds, multi-agent transitions, generative-model helpers, strict experiment
configuration, persisted diagnostics, visualisation, and fail-closed
validation.

[![CI](https://github.com/ActiveInferenceInstitute/ActiveBlockference/actions/workflows/ci.yml/badge.svg)](https://github.com/ActiveInferenceInstitute/ActiveBlockference/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](../LICENSE)

This file is the GitHub-facing signpost. The source repository is the
authoritative record; generated run outputs and notebook execution artefacts
are intentionally not committed.

## Start here

Choose the shortest path for your goal:

| Goal | Entry point |
| --- | --- |
| Install and run the library | [`README.md`](../README.md) |
| Understand the public API | [`docs/api.md`](../docs/api.md) |
| Run a first experiment | [`docs/tutorials/01_single_agent.md`](../docs/tutorials/01_single_agent.md) |
| Configure a reproducible pipeline | [`docs/pipeline.md`](../docs/pipeline.md) and [`configs/README.md`](../configs/README.md) |
| Understand the model and coordinates | [`docs/theory.md`](../docs/theory.md) and [`docs/architecture.md`](../docs/architecture.md) |
| Explore executable examples | [`notebooks/README.md`](../notebooks/README.md) |
| Read the publication source | [`docs/manuscript/README.md`](../docs/manuscript/README.md) |
| Contribute or report a bug | [`docs/contributing.md`](../docs/contributing.md) |
| Report a security issue | [`SECURITY.md`](../SECURITY.md) |

## Quick start

Requires Python 3.10 or newer and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --locked --extra dev
uv run blockference pipeline \
  --config configs/smoke.yml \
  --output-root output \
  --run-name first_run
uv run blockference validation --run-dir output/first_run
```

The pipeline writes a self-contained run directory containing configuration,
trajectory and diagnostic tables, model and policy data, visualisations,
logs, a manifest, and an aggregate validation report. Validation certifies
software and artefact integrity; it does not establish that a scientific
hypothesis is empirically adequate.

For a small Python API example, see the root README. For multi-agent runs,
provide distinct `simulation.initial_states`; the simultaneous-transition
environment rejects ambiguous starting positions.

## What the package provides

* `GridWorld` implements bounded `(y, x)` coordinates, observations,
  simultaneous actions, collision resolution, and serialisation.
* `ActiveGridference` constructs the discrete generative-model components and
  expected-free-energy policy machinery.
* `ExperimentConfig` loads strict YAML/TOML configuration and rejects unknown
  keys, invalid paths, malformed distributions, and invalid agent starts.
* `run_pipeline` runs seeded experiments, persists outputs atomically, renders
  diagnostics, and returns a single `PipelineResult.ok` release verdict.
* The radCAD and cadCAD adapters share the same transition vocabulary, making
  backend comparisons about values rather than backend-specific formatting.

The core package is self-contained. Install the optional `pymdp` extra only
for the `BlockferenceAgent` adapter, and the `research` extra only for the
explicitly selected OpenAI GRTs provider. Offline tests and the release gate
do not require either extra.

## Verification and CI

The GitHub Actions workflow runs on pushes and pull requests targeting
`main`, using the committed `uv.lock` on Python 3.10, 3.11, and 3.12. Each
matrix job runs the canonical release gate, which covers:

* Ruff linting and repository-wide Markdown link/anchor validation;
* the full test suite with a 90% `blockference` coverage floor;
* mypy type checking;
* notebook hygiene and execution;
* real pipeline, validation, and standalone `run` smoke checks;
* manuscript source validation, build, and generated-output drift detection;
* source distribution/wheel construction and clean-environment import smoke.

Run the same gate locally with:

```bash
uv run python scripts/release_check.py
```

The gate is intentionally fail-closed: a required check is not converted into
a skip when an optional backend or generated artefact is unavailable. See
[`docs/development.md`](../docs/development.md) for the development workflow
and [`docs/design-contract.md`](../docs/design-contract.md) for the project
scope and invariants.

## Repository map

```text
blockference/  installable library, CLI, environments, persistence, and viz
configs/       strict examples and smoke configuration
docs/          concepts, API, tutorials, development, and manuscript source
notebooks/     executable teaching examples
GRTs/          optional provider-based local research workflow
tests/         unit, integration, artefact, publication, and notebook checks
scripts/       release, documentation, notebook, and manuscript validators
```

## Scope and limitations

ActiveBlockference is a discrete, CPU-oriented research and teaching library.
It is not a neural-network framework, a learned-embedding system, or a claim
that a passing software validation report proves a model of the world. The
formal assumptions, supported backends, determinism guarantees, and known
limitations are documented in the design contract and manuscript.

The project is released under the [MIT License](../LICENSE). Contributions
should follow [`docs/contributing.md`](../docs/contributing.md); security
reports should use the private GitHub advisory path described in
[`SECURITY.md`](../SECURITY.md).
