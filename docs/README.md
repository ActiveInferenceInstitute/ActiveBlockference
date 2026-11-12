# Documentation

Start here if you are new to the repository:

1. [`../README.md`](../README.md) — installation, quick start, public API, and
   repository map.
2. [`documentation-index.md`](documentation-index.md) — flat index of every
   long-lived document.
3. [`tutorials/`](tutorials/) — short runnable examples.
4. [`api.md`](api.md) — public Python and CLI contracts.
5. [`pipeline.md`](pipeline.md) — configuration, output tree, and validation.

Generated facts live under [`_generated/`](_generated/README.md). Measured gate
numbers are in [`_generated/gate-metrics.md`](_generated/gate-metrics.md); do
not copy those numbers into prose.

## Concepts and design

* [`theory.md`](theory.md) — the discrete Active Inference loop and matrix
  conventions.
* [`architecture.md`](architecture.md) — module boundaries and data flow.
* [`gnn.md`](gnn.md) — the GNN declaration of the grid-world model, the pinned
  parser validation, and its commands.
* [`design-contract.md`](design-contract.md) — scope, constraints, and release
  gates.
* [`glossary.md`](glossary.md) — recurring terms and symbols.
* [`migration-v1.md`](migration-v1.md) — compatibility and upgrade boundaries.

## Working in the repository

* [`development.md`](development.md) — locked development and verification
  commands.
* [`gate.md`](gate.md) — gate arms, CI job map, and where the measured
  numbers live.
* [`contributing.md`](contributing.md) — contribution flow and issue reporting.
* [`../configs/README.md`](../configs/README.md) — configuration examples and
  schema notes.
* [`../notebooks/README.md`](../notebooks/README.md) — executable notebook
  examples.

## Publication source

* [`manuscript/`](manuscript/) — the ordered publication source, formal
  definitions, numbered equations, figures, bibliography, and reproducible
  build commands.

The software verdicts described throughout these documents certify software and
artefact integrity only; they do not establish empirical adequacy of a
scientific hypothesis.
